# Originated from: https://github.com/pytorch/examples/blob/main/mnist/main.py
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from self_projection import SelfProjection

# Evaluations and results:
#
# Standard Conditions:
# python eval_mnist.py --seed=1 --p-size=8 --dropout-rate-i=0.0 --dropout-rate-p=0.25 --batch-size=64 --epochs=10 --lr=1.0 --gamma=0.7
# Results:
# Test set: Average loss: 0.1306, Accuracy: 9598/10000 (96%)
#
# Heavy Reduction with High Dropout:
# python eval_mnist.py --seed=1 --p-size=4 --dropout-rate-i=0.0 --dropout-rate-p=0.75 --batch-size=64 --epochs=10 --lr=1.0 --gamma=0.7
# Results:
# Test set: Average loss: 0.6192, Accuracy: 8230/10000 (82%)
#
# Heavy Reduction with High Dropout of Projection and Extreme Dropout of Input:
# python eval_mnist.py --seed=1 --p-size=4 --dropout-rate-i=0.9 --dropout-rate-p=0.75 --batch-size=64 --epochs=10 --lr=1.0 --gamma=0.7
# Results:
# Test set: Average loss: 1.0966, Accuracy: 7212/10000 (72%)
#
# Total number of trainable parameters: 6090

class Net(nn.Module):
    p_size: int

    def __init__(
        self,
        p_size: int,
        dropout_rate_i: float,
        dropout_rate_p: float,
    ):
        super(Net, self).__init__()

        print("Test-model parameters:")
        print(f"Projection size: {p_size}")
        print(f"Dropout rate input: {dropout_rate_i}")
        print(f"Dropout rate projection: {dropout_rate_p}")

        self.dropout_input = nn.Dropout(dropout_rate_i)
        self.self_projection = SelfProjection(
            size_input=(28, 28),
            size_projection=p_size,
        )
        self.dropout_projection = nn.Dropout(dropout_rate_p)
        self.activation = nn.Tanh()
        self.linear_consolidate = nn.Linear(p_size**2, p_size**2)
        self.linear_interpretate = nn.Linear(p_size**2, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)
        pass

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = x.squeeze(1)
        x = self.dropout_input(x)
        x = self.self_projection(x)[0]
        x = self.activation(x)
        x = self.dropout_projection(x)
        x = x.flatten(1)
        x = self.linear_consolidate(x)
        x = self.activation(x)
        x = self.linear_interpretate(x)
        x = self.log_softmax(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="SelfProjection MNIST evaluation")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--p-size",
        type=int,
        default=4,
        help="SelfProjection size",
    )
    parser.add_argument(
        "--dropout-rate-i",
        type=float,
        default=0.90,
        help="Input dropout rate",
    )
    parser.add_argument(
        "--dropout-rate-p",
        type=float,
        default=0.75,
        help="Projection dropout rate",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("./data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("./data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net(
        p_size=args.p_size,
        dropout_rate_i=args.dropout_rate_i,
        dropout_rate_p=args.dropout_rate_p,
    ).to(device)

    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total number of trainable parameters: {total_trainable_params}")

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "model_mnist.pt")


if __name__ == "__main__":
    main()

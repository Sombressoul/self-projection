# Originated from: https://github.com/pytorch/examples/blob/main/mnist/main.py
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

from modules.self_projection import SelfProjection

# Experimental model.
#
# python eval_cifar100.py --seed=1 --batch-size=64 --epochs=10 --lr=0.001 --wd=0.00001 --gamma=1.0 --model=-1 --sp-depth=4
class ExperimentalModel(nn.Module):
    def __init__(
        self,
        depth: int = 1,
    ):
        super(ExperimentalModel, self).__init__()
        self.self_projection = SelfProjection(
            size_input=(96, 32),
            size_projection=16,
            depth=depth,
        )
        self.lnorm = nn.LayerNorm(16**2)
        self.fc = nn.Linear(16**2, 100)
        self.log_softmax = nn.LogSoftmax(dim=1)
        pass

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = x.view([-1, 96, 32])
        x = self.self_projection(x)
        x = x.flatten(1)
        x = self.lnorm(x)
        x = self.fc(x)
        x = self.log_softmax(x)
        return x


# Model for evaluation: SelfProjection
#
# SelfProjection depth -> 1
# python eval_cifar100.py --seed=1 --batch-size=64 --epochs=10 --lr=0.001 --wd=0.00001 --gamma=1.0 --model=0 --sp-depth=1
#
# SelfProjection depth -> 4
# python eval_cifar100.py --seed=1 --batch-size=64 --epochs=10 --lr=0.001 --wd=0.00001 --gamma=1.0 --model=0 --sp-depth=4
class NetSP(nn.Module):
    def __init__(
        self,
        depth: int = 1,
    ):
        super(NetSP, self).__init__()
        self.self_projection = SelfProjection(
            size_input=(96, 32),
            size_projection=16,
            depth=depth,
        )
        self.activation = nn.ReLU()
        self.fc = nn.Linear(16**2, 100)
        self.log_softmax = nn.LogSoftmax(dim=1)
        pass

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = x.view([-1, 96, 32])
        x = self.self_projection(x)
        x = self.activation(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.log_softmax(x)
        return x


# Model for evaluation: reference
# python eval_cifar100.py --seed=1 --batch-size=64 --epochs=10 --lr=0.001 --wd=0.00001 --gamma=1.0 --model=1
class NetCNN(nn.Module):
    def __init__(
        self,
    ):
        super(NetCNN, self).__init__()
        self.conv_a = nn.Conv2d(3, 16, 3, 1, 1)
        self.pooling_a = nn.MaxPool2d(2, 2)
        self.conv_b = nn.Conv2d(16, 32, 3, 1, 1)
        self.pooling_b = nn.MaxPool2d(2, 2)
        self.conv_reductor = nn.Conv2d(32, 4, 1, 1, 0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 8 * 8, 100)
        self.log_softmax = nn.LogSoftmax(dim=1)
        pass

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = self.conv_a(x)
        x = self.relu(x)
        x = self.pooling_a(x)
        x = self.conv_b(x)
        x = self.relu(x)
        x = self.pooling_b(x)
        x = self.conv_reductor(x)
        x = x.flatten(1)
        x = self.fc(x)
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
    parser = argparse.ArgumentParser(description="SelfProjection CIFAR-100 evaluation")
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
        default=1.0e-3,
        metavar="LR",
        help="learning rate (default: 1.0e-3)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1.0e-5,
        metavar="WD",
        help="Weight decay (default: 1.0e-5)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        metavar="M",
        help="Learning rate step gamma (default: 1.0 - constant)",
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
        "--model",
        type=int,
        default=0,
        help="model type: 0 - SP, 1 - CNN, -1 - experimental (default: 0)",
    )
    parser.add_argument(
        "--sp-depth",
        type=int,
        default=1,
        help="SelfProjection depth (default: 1)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
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

    transform = transforms.Compose([transforms.ToTensor()])
    dataset1 = datasets.CIFAR100(
        "./data", train=True, download=True, transform=transform
    )
    dataset2 = datasets.CIFAR100("./data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    if args.model == 0:
        model = NetSP(depth=args.sp_depth).to(device)
    elif args.model == 1:
        model = NetCNN().to(device)
    elif args.model == -1:
        model = ExperimentalModel().to(device)
    else:
        raise "Unknown model type."

    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total number of trainable parameters: {total_trainable_params}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "model_cifar100.pt")


if __name__ == "__main__":
    main()

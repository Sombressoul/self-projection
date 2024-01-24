import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from self_projection import (
    SelfProjectionDev,
    SelfProjection,
)

# seeding
torch.manual_seed(2)

# test params
input_size = 16
projection_size = 8
matrices_count = 2048
epochs = 1.0e4
log_loss_nth_epoch = 100
lr = 1.0e-3
wd = 1.0e-4

plot_loss = True

dev_mode = False
args_dev = dict(
    depth=4,
    preserve_distribution=False,
    standardize_output=False,
    scale_and_bias=False,
)
args_prod = dict(
    depth=1,
)


class Net(nn.Module):
    def __init__(
        self,
        i_size: int,
        p_size: int,
        **kwargs,
    ):
        super(Net, self).__init__()

        global dev_mode

        sp_args = args_dev if dev_mode else args_prod
        sp_class = SelfProjection if not dev_mode else SelfProjectionDev

        print(f"Using: {sp_class.__name__}")

        self.self_projection_encode = sp_class(
            size_input=[i_size] * 2,
            size_projection=p_size,
            **sp_args,
        )
        self.self_projection_decode = sp_class(
            size_input=[p_size] * 2,
            size_projection=i_size,
            **sp_args,
        )
        pass

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = (x - x.mean(dim=[-1, -2], keepdim=True)) / (x.std() + 1.0e-5)
        x = self.self_projection_encode(x)
        x = self.self_projection_decode(x)
        return x


def train(model, optimizer, mat_input, mat_target, epochs):
    loss_accumulator = []
    for epoch in range(int(epochs)):
        optimizer.zero_grad()
        matrices_output = model(mat_input)
        loss = F.mse_loss(matrices_output, mat_target)
        loss.backward()
        optimizer.step()

        if epoch > 0 and epoch % log_loss_nth_epoch == 0:
            print(f"epoch {epoch} loss {loss.item()}")

        loss_accumulator.append(loss.item())

    if plot_loss:
        _plot_loss(loss_accumulator)

    pass


def _plot_loss(loss_accumulator):
    dpi = 100
    w_inches = 1200 / dpi
    h_inches = 1200 / dpi

    final_loss = loss_accumulator[-1]
    min_loss = min(loss_accumulator)
    max_loss = max(loss_accumulator)
    epoch_count = len(loss_accumulator)
    min_loss_idx = loss_accumulator.index(min_loss)
    last_10_percent = loss_accumulator[int(0.9 * epoch_count) :]
    last_10_mean = sum(last_10_percent) / len(last_10_percent)
    last_10_std = (
        sum([(x - last_10_mean) ** 2 for x in last_10_percent]) / len(last_10_percent)
    ) ** 0.5

    get_y = lambda x: (max_loss - min_loss) * x + min_loss

    plt.rcParams["figure.dpi"] = dpi
    plt.figure(figsize=(w_inches, h_inches))
    plt.plot(loss_accumulator)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.annotate(
        f"Max Loss: {max_loss:.4f}",
        xy=(0, max_loss),
        xytext=(epoch_count / 2, get_y(1.0)),
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=0.5),
    )
    plt.annotate(
        f"Final Loss: {final_loss:.4f}",
        xy=(epoch_count - 1, final_loss),
        xytext=(epoch_count / 2, get_y(0.8)),
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=0.5),
    )
    plt.annotate(
        f"Min Loss: {min_loss:.4f}",
        xy=(min_loss_idx, min_loss),
        xytext=(epoch_count / 2, get_y(0.6)),
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=0.5),
    )
    plt.figtext(
        0.5,
        0.01,
        f"Mean (last 10%): {last_10_mean:.4f}, Std (last 10%): {last_10_std:.4f}",
        ha="center",
        fontsize=10,
    )

    # Show plot
    plt.show()


# train
model = Net(input_size, projection_size).to("cuda")
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

matrices_input = torch.rand([matrices_count, *([input_size] * 2)]).to(
    dtype=torch.float32, device="cuda"
)
matrices_target = matrices_input.clone()

train(
    model,
    optimizer,
    matrices_input,
    matrices_target,
    epochs,
)

# test
matrices_output = model(matrices_input)

print(f"Target 1-st:")
print(matrices_target[0, 0, 0:16].cpu().detach().numpy().tolist())
print(f"Output 1-st:")
print(matrices_output[0, 0, 0:16].cpu().detach().numpy().tolist())

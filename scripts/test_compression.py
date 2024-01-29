import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

from self_projection import SelfProjection
from self_projection.utils.functional import plot_loss as _plot_loss

# seeding
torch.manual_seed(1)

# test params
input_size = 16
projection_size = 8
matrices_count = 2048
epochs = 1.0e4
log_loss_nth_epoch = 100
lr = 1.0e-4
wd = 1.5e-5

plot_loss = True

dev_mode = True
args_dev = dict(
    depth=1,
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

        self.self_projection_encode = SelfProjection(
            size_input=[i_size] * 2,
            size_projection=p_size,
            **sp_args,
        )
        self.self_projection_decode = SelfProjection(
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

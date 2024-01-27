import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

from modules.self_projection import SelfProjection

# seeding
torch.manual_seed(1)

depth = 1
input_shape = [96, 16]
matrices_count = 128
matrices_input = torch.rand([matrices_count, *input_shape]).to(
    dtype=torch.float32, device="cuda"
)
projection_size = 16
matrices_target = torch.rand([matrices_count, *([projection_size] * 2)]).to(
    dtype=torch.float32, device="cuda"
)
epochs = 1.0e4


class Net(nn.Module):
    def __init__(
        self,
        i_size: list[int],
        p_size: int,
    ):
        super(Net, self).__init__()
        self.self_projection_a = SelfProjection(
            size_input=i_size,
            size_projection=p_size,
            depth=depth,
        )
        self.self_projection_b = SelfProjection(
            size_input=[p_size, p_size],
            size_projection=p_size,
            depth=depth,
        )
        pass

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = (x - x.mean(dim=[-1, -2], keepdim=True)) / (x.std() + 1.0e-5)
        x = self.self_projection_a(x)
        x = self.self_projection_b(x)
        return x


def train(model, optimizer, mat_input, mat_target, epochs):
    for epoch in range(int(epochs)):
        optimizer.zero_grad()
        matrices_output = model(mat_input)
        loss = F.mse_loss(matrices_output, mat_target)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch {epoch} loss {loss.item()}")


# train
model = Net(input_shape, projection_size).to("cuda")
optimizer = optim.Adam(model.parameters(), lr=1.0e-3)

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
print(matrices_target[0])
print(f"Output 1-st:")
print(matrices_output[0])

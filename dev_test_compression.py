import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from self_projection import SelfProjection

# seeding
torch.manual_seed(2)

input_size = 128
projection_size = 64
matrices_count = 256
epochs = 1.0e4
depth = 1


class Net(nn.Module):
    def __init__(
        self,
        i_size: int,
        p_size: int,
    ):
        super(Net, self).__init__()
        self.self_projection_encode = SelfProjection(
            size_input=[i_size] * 2,
            size_projection=p_size,
            depth=depth,
        )
        self.self_projection_decode = SelfProjection(
            size_input=[p_size] * 2,
            size_projection=i_size,
            depth=depth,
        )
        pass

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = (x - x.mean(dim=[-1, -2], keepdim=True)) / (x.std() + 1.0e-5)
        x = self.self_projection_encode(x)[0]
        x = self.self_projection_decode(x)[0]
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
model = Net(input_size, projection_size).to("cuda")
optimizer = optim.Adam(model.parameters(), lr=1.0e-3)

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
print(matrices_target[0])
print(f"Output 1-st:")
print(matrices_output[0])

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from madgrad import MADGRAD

from self_projection import SelfProjection

# seeding
torch.manual_seed(42)

# AE params
self_projection_depth = 8
epochs = 100
folder_path = "data/ae_test_5k"
batch_size = 32
dropout_rate = 0.25
log_nth_epoch = 1
log_image_idx = 0


class Net(nn.Module):
    def __init__(
        self,
        i_size: int,
        depth: int,
        dropout_rate: float,
    ):
        super(Net, self).__init__()
        self.layer_norm_input = nn.LayerNorm([i_size] * 2)
        self.self_projection_encode_a = SelfProjection(
            size_input=[i_size] * 2,
            size_projection=128,
            depth=depth,
        )
        self.layer_norm_encode_a = nn.LayerNorm([128] * 2)
        self.self_projection_encode_b = SelfProjection(
            size_input=[128] * 2,
            size_projection=64,
            depth=depth,
        )
        self.layer_norm_encode_b = nn.LayerNorm([64] * 2)
        self.self_projection_encode_c = SelfProjection(
            size_input=[64] * 2,
            size_projection=32,
            depth=depth,
        )
        self.layer_norm_encode_c = nn.LayerNorm([32] * 2)
        self.self_projection_decode_c = SelfProjection(
            size_input=[32] * 2,
            size_projection=64,
            depth=depth,
        )
        self.layer_norm_decode_c = nn.LayerNorm([64] * 2)
        self.self_projection_decode_b = SelfProjection(
            size_input=[64] * 2,
            size_projection=128,
            depth=depth,
        )
        self.layer_norm_decode_b = nn.LayerNorm([128] * 2)
        self.self_projection_decode_a = SelfProjection(
            size_input=[128] * 2,
            size_projection=i_size,
            depth=depth,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        pass

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = self.layer_norm_input(x)
        x = self.self_projection_encode_a(x)[0]
        x = self.activation(x)
        x = self.layer_norm_encode_a(x)
        x = self.self_projection_encode_b(x)[0]
        x = self.activation(x)
        x = self.layer_norm_encode_b(x)
        x = self.self_projection_encode_c(x)[0]
        x = self.activation(x)
        x = self.layer_norm_encode_c(x)
        latents = x.clone()
        x = self.self_projection_decode_c(x)[0]
        x = self.activation(x)
        x = self.layer_norm_decode_c(x)
        x = self.self_projection_decode_b(x)[0]
        x = self.activation(x)
        x = self.layer_norm_decode_b(x)
        x = self.self_projection_decode_a(x)[0]
        return x, latents


def train(model, optimizer, tensor_images, epochs):
    global log_nth_epoch, log_image_idx
    loss_data = []
    for epoch in tqdm.tqdm(range(int(epochs))):
        # batch_size
        idxs = torch.randperm(tensor_images.shape[0])
        tensor_images = tensor_images[idxs]

        running_loss = 0.0

        # inner tqdm
        for i in tqdm.tqdm(range(0, tensor_images.shape[0], batch_size)):
            # for i in range(0, tensor_images.shape[0], batch_size):
            optimizer.zero_grad()
            batch = tensor_images[i : i + batch_size]
            matrices_output, latents = model(batch)

            loss = F.mse_loss(matrices_output, batch)

            # loss = F.huber_loss(matrices_output, batch, reduction="mean", delta=0.75)

            # x = F.log_softmax(matrices_output, dim=-2)
            # y = F.log_softmax(batch, dim=-2)
            # loss = F.kl_div(x, y, reduction="batchmean", log_target=True)

            loss.backward()
            running_loss += loss.item() * batch.shape[0]
            optimizer.step()

        running_loss = running_loss / tensor_images.shape[0]

        if epoch % log_nth_epoch == 0:
            size = [matrices_output.shape[-2], matrices_output.shape[-1]]
            latents = latents[log_image_idx]
            latents = latents.reshape([1, 1, *latents.shape])
            latents = (latents - latents.min()) / (latents.max() - latents.min())
            latents = F.interpolate(latents, size=size, mode="bilinear")
            latents = latents.reshape([latents.shape[-2], latents.shape[-1]])
            combined_tensor = torch.cat(
                (batch[log_image_idx], matrices_output[log_image_idx], latents), 1
            ).clip(0.0, 1.0)
            combined_image = tensor_to_image(combined_tensor)
            text = f"Loss: {running_loss:.4f}"
            combined_image = add_text_to_image(combined_image, text)
            combined_image.save(f"temp/epoch_{str(epoch)}.png")
            # print(f"epoch {epoch} loss {running_loss}")
            loss_data.append(running_loss)

    plt.plot(loss_data)
    plt.show()


def load_images_from_folder(folder, transform=None):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Add/check file formats
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                if transform:
                    img = transform(img)
                images.append(img)
    return images


def tensor_to_image(tensor):
    image = transforms.ToPILImage()(tensor)
    image = image.convert("RGB")
    return image


def add_text_to_image(image, text, position=(10, 10), font_size=20, font_color="red"):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(font_size)
    draw.text(position, text, fill=font_color, font=font)
    return image


# Prepare images from folder
transform = transforms.Compose(
    [
        transforms.Grayscale(),  # Converts to grayscale
        transforms.ToTensor(),  # Converts to tensor
    ]
)

images = load_images_from_folder(folder_path, transform)
tensor_images = torch.stack(images).squeeze(1).to(dtype=torch.float32, device="cuda")

assert (
    tensor_images.shape[1] == tensor_images.shape[2]
), f"Image shape is not square: {tensor_images.shape}"

tensor_images = tensor_images.clone()

# train
input_size = tensor_images.shape[1]
model = Net(input_size, self_projection_depth, dropout_rate).to("cuda")
optimizer = MADGRAD(
    model.parameters(),
    lr=1.0e-3,
    momentum=0.9,
    weight_decay=0.0,
)

total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_trainable_params}")

train(
    model,
    optimizer,
    tensor_images,
    epochs,
)

torch.save(model.state_dict(), "temp/model_autoencoder.pt")

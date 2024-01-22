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

from models import SimpleAutoencoderSPSP

# seeding
torch.manual_seed(42)

# AE params
model_dev_mode = True
epochs = 100
folder_path = "data/ae_test_5k_2_1"
batch_size = 64
log_nth_epoch = 1
log_image_idx = 0
save_model_nth_epoch = 10
save_image_nth_batch = 0

use_clip_grad_value = False
clip_grad_value = 1.0

use_clip_grad_norm = True
clip_grad_norm_max = 0.5
clip_grad_norm_type = 2.0

model_sp_depth = 4
model_nn_depth = 3

load_from_checkpoint = True
checkpoint_path = "temp/AutoencoderSPSP_02.pt"


def train(model, optimizer, tensor_images, epochs):
    global log_nth_epoch, log_image_idx
    loss_data = []
    for epoch in tqdm.tqdm(range(int(epochs))):
        idxs = torch.randperm(tensor_images.shape[0])
        tensor_images = tensor_images[idxs]

        running_loss = 0.0

        for i in tqdm.tqdm(range(0, tensor_images.shape[0], batch_size)):
            batch_idx = i // batch_size
            optimizer.zero_grad()
            batch = tensor_images[i : i + batch_size]
            matrices_output, latents = model(batch)

            loss = F.mse_loss(matrices_output, batch)

            # loss = F.huber_loss(matrices_output, batch, reduction="mean", delta=1.0)

            # x = F.log_softmax(matrices_output, dim=-2)
            # y = F.log_softmax(batch, dim=-2)
            # loss = F.kl_div(x, y, reduction="batchmean", log_target=True)

            loss.backward()
            running_loss += loss.item() * batch.shape[0]

            if save_image_nth_batch > 0 and batch_idx % save_image_nth_batch == 0:
                log_image(
                    f"epoch_{str(epoch)}_batch_{str(batch_idx)}",
                    batch[log_image_idx],
                    matrices_output[log_image_idx],
                    latents[log_image_idx],
                    running_loss / (i + batch_size),
                )

            assert not all(
                [use_clip_grad_value, use_clip_grad_norm]
            ), "Only one of clip_grad_value or clip_grad_norm can be used"

            if use_clip_grad_value:
                nn.utils.clip_grad_value_(
                    model.parameters(),
                    clip_value=clip_grad_value,
                )
            if use_clip_grad_norm:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=clip_grad_norm_max,
                    norm_type=clip_grad_norm_type,
                )

            optimizer.step()

        running_loss = running_loss / tensor_images.shape[0]

        if epoch % log_nth_epoch == 0:
            # size = [matrices_output.shape[-2], matrices_output.shape[-1]]
            # latents = latents[log_image_idx]
            # latents = latents.reshape([1, 1, *latents.shape])
            # latents = (latents - latents.min()) / (latents.max() - latents.min())
            # latents = F.interpolate(latents, size=size, mode="nearest")
            # latents = latents.reshape([latents.shape[-2], latents.shape[-1]])
            # combined_tensor = torch.cat(
            #     (batch[log_image_idx], matrices_output[log_image_idx], latents), 1
            # ).clip(0.0, 1.0)
            # combined_image = tensor_to_image(combined_tensor)
            # text = f"Loss: {running_loss:.4f}"
            # combined_image = add_text_to_image(combined_image, text)
            # combined_image.save(f"temp/epoch_{str(epoch)}.png")
            log_image(
                f"epoch_{str(epoch)}",
                batch[log_image_idx],
                matrices_output[log_image_idx],
                latents[log_image_idx],
                running_loss,
            )
            # print(f"epoch {epoch} loss {running_loss}")
            loss_data.append(running_loss)

        if epoch % save_model_nth_epoch == 0:
            torch.save(model.state_dict(), f"temp/model_epoch_{str(epoch)}.pth")

    plt.plot(loss_data)
    plt.show()


def log_image(name, t_src_image, t_output_image, t_latents, f_loss):
    size = [t_src_image.shape[-2], t_src_image.shape[-1]]

    t_latents = t_latents.reshape([1, 1, *t_latents.shape])
    t_latents = (t_latents - t_latents.min()) / (t_latents.max() - t_latents.min())
    t_latents = F.interpolate(t_latents, size=size, mode="nearest")
    t_latents = t_latents.reshape([t_latents.shape[-2], t_latents.shape[-1]])

    combined_tensor = torch.cat((t_src_image, t_output_image, t_latents), 1).clip(
        0.0, 1.0
    )
    combined_image = tensor_to_image(combined_tensor)

    text = f"Loss: {f_loss:.4f}"
    combined_image = add_text_to_image(combined_image, text)
    combined_image.save(f"temp/{str(name)}.png")

    pass


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

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model

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
model = SimpleAutoencoderSPSP(
    input_size=input_size,
    network_depth=model_nn_depth,
    self_projection_depth=model_sp_depth,
    dev=model_dev_mode,
).to("cuda")

if load_from_checkpoint:
    model = load_model(model, checkpoint_path)

optimizer = MADGRAD(
    model.parameters(),
    lr=1.0e-4,
    momentum=0.9,
    weight_decay=0.0,
)
# optimizer = optim.Adam(
#     model.parameters(),
#     lr=1.0e-5,
#     weight_decay=1.0e-4,
# )
# optimizer = optim.SGD(
#     model.parameters(),
#     lr=1.0e-3,
# )

total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_trainable_params}")

train(
    model,
    optimizer,
    tensor_images,
    epochs,
)

torch.save(model.state_dict(), "temp/model_autoencoder.pt")

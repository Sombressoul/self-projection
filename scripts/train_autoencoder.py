import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

# ================================================================================= #
# ____________________________> Module imports.
# ================================================================================= #
# Add parent dir to path to enable imports from the module.
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

from modules.models import (
    SimpleAutoencoderSPSP,
    SimpleAutoencoderSPSingle,
    AutoencoderCNNSP,
)

# ================================================================================= #
# ____________________________> Config.
# ================================================================================= #
# Seeding:
torch.manual_seed(1)

# Data:
images_path = "data/autoencoder_cnnsp"
training_log_path = "models/train"

# Training:
epochs = 100
batch_size = 32

# Model:
model_class = AutoencoderCNNSP
model_nn_depth = 1
debug_model = False

# Class dependent:
AutoencoderCNNSP_scale_factor = 8
AutoencoderCNNSP_channels_base = 16
AutoencoderCNNSP_extractor_depth = 1
AutoencoderCNNSP_compressor_depth = 1
AutoencoderCNNSP_use_extractor = True
AutoencoderCNNSP_use_compressor = True
AutoencoderCNNSP_baseline = False
AutoencoderCNNSP_dropout_rate = 0.05

# Optimization:
warmup_steps = 256
learning_rate_start = 1.0e-9
learning_rate_end = 1.5e-4
weight_decay = 1.0e-2
use_amsgrad = True
use_clip_grad_value = False
clip_grad_value = 1.0
use_clip_grad_norm = False
clip_grad_norm_max = 1.0
clip_grad_norm_type = 2.0  # L2
use_clip_grad_n_sched = True
clip_grad_n_sched_min = 1.0e-3
clip_grad_n_sched_max = 1.5
clip_grad_n_sched_steps = 256
clip_grad_n_sched_type = 2.0

# Logging/Plotting:
log_nth_epoch = 1
save_image_nth_batch = 0
save_image_nth_epoch = 1
plot_results = False

# Checkpointing:
dtype = torch.bfloat16
save_model = True
save_model_nth_epoch = 10
load_from_checkpoint = True
checkpoint_path = "models/train"
checkpoint_name = "AutoencoderCNNSP_epoch_40"

# SelfProjection:
sp_params = dict(
    depth=1,
    preserve_distribution=False,
    standardize_output=False,
    scale_and_bias=False,
)


# ================================================================================= #
# ____________________________> Helper functions.
# ================================================================================= #
def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    batch_size: int,
    tensor_images: torch.Tensor,
    epochs: int,
    log_nth_epoch: int,
):
    total_trainable_params = get_trainable_params_cnt(model)
    print(f"Model class: {model.__class__.__name__}")
    print(f"Total number of trainable parameters: {total_trainable_params}")

    loss_data = []
    for epoch in tqdm.tqdm(range(int(epochs))):
        idxs = torch.randperm(tensor_images.shape[0])
        tensor_images = tensor_images[idxs]

        running_loss = 0.0

        for i in tqdm.tqdm(range(0, tensor_images.shape[0], batch_size)):
            batch_idx = i // batch_size
            optimizer.zero_grad()
            targets = tensor_images[i : i + batch_size]
            outputs, latents = model(targets)
            loss = F.l1_loss(
                input=outputs,
                target=targets,
            )
            loss.backward()
            running_loss += loss.item() * targets.shape[0]

            if save_image_nth_batch > 0 and batch_idx % save_image_nth_batch == 0:
                log_image(
                    name=f"{model.__class__.__name__}_epoch_{str(epoch)}_batch_{str(batch_idx)}",
                    image_target=targets[0],
                    image_output=outputs[0],
                    latents=latents[0],
                    loss=running_loss / (i + batch_size),
                    path=training_log_path,
                )

            assert (
                sum(
                    [
                        int(use_clip_grad_value),
                        int(use_clip_grad_norm),
                        int(use_clip_grad_n_sched),
                    ]
                )
                < 2
            ), "".join(
                [
                    "Only one of the following options can be used: ",
                    "use_clip_grad_value, use_clip_grad_norm, ",
                    "use_clip_grad_norm_scheduled",
                ]
            )

            if use_clip_grad_value:
                nn.utils.clip_grad_value_(
                    parameters=model.parameters(),
                    clip_value=clip_grad_value,
                )
            if use_clip_grad_norm:
                nn.utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=clip_grad_norm_max,
                    norm_type=clip_grad_norm_type,
                )
            if use_clip_grad_n_sched:
                try:
                    clip_grad_n_sched_val = clip_grad_n_sched_val + clip_grad_n_step
                    clip_grad_n_sched_val = min(
                        clip_grad_n_sched_val,
                        clip_grad_n_sched_max,
                    )
                except NameError:
                    clip_grad_n_step = (
                        clip_grad_n_sched_max - clip_grad_n_sched_min
                    ) / clip_grad_n_sched_steps
                    clip_grad_n_sched_val = clip_grad_n_sched_min

                nn.utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=clip_grad_n_sched_val,
                    norm_type=clip_grad_n_sched_type,
                )

            optimizer.step()
            scheduler.step()

        running_loss = running_loss / tensor_images.shape[0]

        if epoch % log_nth_epoch == 0:
            loss_data.append(running_loss)

        if save_image_nth_epoch > 0 and epoch % save_image_nth_epoch == 0:
            log_image(
                name=f"{model.__class__.__name__}_epoch_{str(epoch)}",
                image_target=targets[0],
                image_output=outputs[0],
                latents=latents[0],
                loss=running_loss,
                path=training_log_path,
            )

        if save_model and epoch % save_model_nth_epoch == 0:
            model_save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                path=checkpoint_path,
                name=f"{model.__class__.__name__}_epoch_{str(epoch)}",
            )

    if plot_results:
        plt.plot(loss_data)
        plt.show()


def log_image(
    name: str,
    image_target: torch.Tensor,
    image_output: torch.Tensor,
    latents: torch.Tensor,
    loss: float,
    path: str,
) -> None:
    size = [image_target.shape[-2], image_target.shape[-1]]

    latents = latents.reshape([1, 1, *latents.shape])
    latents = (latents - latents.min()) / (latents.max() - latents.min())
    latents = F.interpolate(latents, size=size, mode="nearest")
    latents = latents.reshape([latents.shape[-2], latents.shape[-1]])

    combined_tensor = torch.cat((image_target, image_output, latents), 1).clip(0.0, 1.0)
    combined_image = tensor_to_image(combined_tensor)

    text = f"Loss: {loss:.4f}"
    combined_image = add_text_to_image(combined_image, text)
    combined_image.save(f"{path}/{str(name)}.png")

    pass


def load_images_from_folder(
    folder: str,
    transform: transforms.Compose = None,
) -> list[Image.Image]:
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Add/check file formats
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                if transform:
                    img = transform(img)
                images.append(img)
    return images


def tensor_to_image(
    tensor: torch.Tensor,
) -> Image.Image:
    image = transforms.ToPILImage()(tensor)
    image = image.convert("RGB")
    return image


def add_text_to_image(
    image: Image.Image,
    text: str,
    position: tuple[int, int] = (10, 10),
    font_size: int = 20,
    font_color: str = "red",
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(font_size)
    draw.text(position, text, fill=font_color, font=font)
    return image


def model_load(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    path: str,
    name: str,
) -> tuple[nn.Module, torch.optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    model.load_state_dict(torch.load(os.path.join(path, f"{name}.pt")))
    optimizer.load_state_dict(torch.load(os.path.join(path, f"{name}.opt.pt")))
    scheduler.load_state_dict(torch.load(os.path.join(path, f"{name}.sch.pt")))
    return model, optimizer, scheduler


def model_save(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    path: str,
    name: str,
) -> None:
    torch.save(model.state_dict(), os.path.join(path, f"{name}.pt"))
    torch.save(optimizer.state_dict(), os.path.join(path, f"{name}.opt.pt"))
    torch.save(scheduler.state_dict(), os.path.join(path, f"{name}.sch.pt"))
    pass


def get_trainable_params_cnt(
    model: nn.Module,
) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ================================================================================= #
# ____________________________> Entry point.
# ================================================================================= #
if __name__ == "__main__":
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    images = load_images_from_folder(images_path, transform)
    tensor_images = torch.stack(images).squeeze(1).to(dtype=dtype, device="cuda")

    assert (
        tensor_images.shape[1] == tensor_images.shape[2]
    ), f"Image shape is not square: {tensor_images.shape}"

    if model_class == SimpleAutoencoderSPSP:
        model = (
            SimpleAutoencoderSPSP(
                input_size=tensor_images.shape[1],
                network_depth=model_nn_depth,
                sp_params=sp_params,
            )
            .to(dtype)
            .to("cuda")
        )
    elif model_class == SimpleAutoencoderSPSingle:
        model = (
            SimpleAutoencoderSPSingle(
                input_size=tensor_images.shape[1],
                sp_params=sp_params,
            )
            .to(dtype)
            .to("cuda")
        )
    elif model_class == AutoencoderCNNSP:
        model = (
            AutoencoderCNNSP(
                input_size=tensor_images.shape[1],
                network_depth=model_nn_depth,
                scale_factor=AutoencoderCNNSP_scale_factor,
                channels_base=AutoencoderCNNSP_channels_base,
                compressor_depth=AutoencoderCNNSP_compressor_depth,
                extractor_depth=AutoencoderCNNSP_extractor_depth,
                use_extractor=AutoencoderCNNSP_use_extractor,
                use_compressor=AutoencoderCNNSP_use_compressor,
                baseline=AutoencoderCNNSP_baseline,
                dropout_rate=AutoencoderCNNSP_dropout_rate,
                sp_params=sp_params,
            )
            .to(dtype)
            .to("cuda")
        )
    else:
        raise Exception(f"Unknown model class: {model_class}")

    if debug_model:
        model_params = get_trainable_params_cnt(model)
        print("DEBUG INFO:")
        print(f"Model class: {model.__class__.__name__}")
        print(f"Total number of trainable parameters: {model_params}")
        print(model)
        exit()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate_start,
        weight_decay=weight_decay,
        amsgrad=use_amsgrad,
    )

    scheduler_warmup = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=warmup_steps,
        eta_min=learning_rate_end,
        verbose=False,
    )
    scheduler_constant = optim.lr_scheduler.ConstantLR(
        optimizer=optimizer,
        factor=1.0,
        total_iters=1,
        verbose=False,
    )
    scheduler_constant.base_lrs = [learning_rate_end] * len(optimizer.param_groups)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[scheduler_warmup, scheduler_constant],
        milestones=[warmup_steps],
    )

    if load_from_checkpoint:
        model, optimizer, scheduler = model_load(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            path=checkpoint_path,
            name=checkpoint_name,
        )
        model = model.to(dtype).to("cuda")

    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=batch_size,
        tensor_images=tensor_images,
        epochs=epochs,
        log_nth_epoch=log_nth_epoch,
    )

    if save_model:
        model_save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            path=checkpoint_path,
            name=f"{model.__class__.__name__}_trained",
        )

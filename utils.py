import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# Converts from LAB to RGB color space
def lab_to_rgb(L_norm, ab_norm):
    # Undo the normalization
    L = L_norm * 255
    a = ab_norm[...,0] * 128.0 + 128.0
    b = ab_norm[...,1] * 128.0 + 128.0

    # Merge the three channels 
    lab = np.stack([L, a, b], axis=-1).astype("uint8")

    # Convert from LAB to RGB and normalize to [0, 1]
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    rgb = np.clip(rgb / 255, 0, 1)
    return rgb

# Plots the generator and discriminator losses over epochs
def plot_gd_losses(g_losses, d_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(g_losses, label="Generator Loss", linewidth=2, color="#B77DD4")
    plt.plot(d_losses, label="Discriminator Loss", linewidth=2, color="#FFB2CC")
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.xlabel("Epoch",)
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss Over Epochs")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plots a validation metric (e.g., L1 loss or PSNR) over epochs
def plot_val_metric(metric_values, metric_name):
    plt.figure(figsize=(10, 6))
    plt.plot(metric_values, linewidth=2, color="hotpink")
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"Validation {metric_name} Over Epochs")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# TODO finish and comment this function 
# def show_image(sample, mode=0):
#     mode 0 is both, 1 is l , 2 is ab


#     L_norm = sample['L'].numpy()[0]                   # (1,224,224) -> (244,244)
#     ab_norm = sample['ab'].numpy().transpose(1,2,0)   # (2,244,244) -> (224,224,2)

#     # Show the grayscale L channel

#     plt.figure(figsize=(10,5))

#     plt.subplot(1,2,1)
#     plt.imshow(L_norm, cmap="gray")
#     plt.title("Grayscale L channel")
#     plt.axis("off")

#     # Convert to RGB
#     rgb = lab_to_rgb(L_norm, ab_norm)

#     # Visualize
#     plt.subplot(1,2,2)
#     plt.imshow(rgb)
#     plt.title("Reconstructed RGB from LAB")
#     plt.axis("off")
#     plt.show()
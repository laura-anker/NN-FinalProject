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

# Take ANY image, crop/resize it, convert to LAB, and produce the correct L tensor
def prepare_ijmage(image_path, target_size=224):
    # load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at: {image_path}")
    
    # convert from OpenCV BRG -> RGB for our model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # extract height, width, # channels in image
    h, w, _ = img.shape
    min_dim = min(h,w)

    # compute how much to crop from each side
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2

    # numpy slicing to extract centered square
    img_cropped = img[top:top+min_dim, left:left+min_dim]

    # resize to model input size
    img_resized = cv2.resize(img_cropped, (target_size, target_size))
    
    # convert to lab
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)

    # extract L channel, normalize, convert to Pytorch tensor
    L = lab[:, :, 0].astype("float32")
    L_norm = L / 255.0
    L_tensor = torch.tensor(L_norm).unsqueeze(0).unsqueeze(0).float()

    return L_tensor, img_resized

# visualize RGB prediction for images outside dataset 
def predict_color(generator, L_tensor):
    # send input tensor to same device as model
    device = next(generator.parameters()).device
    generator.eval()

    # turn off gradient tracking to speed up inference & save memory
    with torch.no_grad():
        L_input = L_tensor.to(device)
        fake_ab = generator(L_input)[0].cpu().permute(1,2,0).numpy()

    # convert normalized → real LAB values
    a = fake_ab[..., 0] * 128 + 128
    b = fake_ab[..., 1] * 128 + 128

    # reconstruct L channel back to 0–255, and stack L,a,b into LAB image
    L = (L_tensor.numpy()[0, 0] * 255).astype("float32")
    lab = np.stack([L, a, b], axis=-1).astype("uint8")

    # convert LAB -> color then normalize
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    rgb = np.clip(rgb / 255.0, 0, 1)

    return rgb
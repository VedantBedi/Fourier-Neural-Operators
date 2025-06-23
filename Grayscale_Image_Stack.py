import torch
import torch.nn as nn
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# =====================================================
#                Spectral Convolution 3D
# =====================================================
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        B, C, D1, D2, D3 = x.shape
        x_ft = torch.fft.rfftn(x, dim=[2, 3, 4])
        out_ft = torch.zeros_like(x_ft, dtype=torch.cfloat)

        m1 = min(self.modes1, x_ft.shape[2])
        m2 = min(self.modes2, x_ft.shape[3])
        m3 = min(self.modes3, x_ft.shape[4])

        out_ft[:, :, :m1, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, :m2, :m3], self.weights[:, :, :m1, :m2, :m3]
        )

        x = torch.fft.irfftn(out_ft, s=(D1, D2, D3), dim=[2, 3, 4])
        return x

# =====================================================
#                       FNO 3D
# =====================================================
class FNO3D(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, out_channels=1):
        super().__init__()
        self.fc0 = nn.Linear(4, width)  # input: 1 channel + 3 coordinates

        self.conv_layers = nn.ModuleList([
            SpectralConv3d(width, width, modes1, modes2, modes3) for _ in range(4)
        ])
        self.w_layers = nn.ModuleList([
            nn.Conv3d(width, width, 1) for _ in range(4)
        ])

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
        for conv, w in zip(self.conv_layers, self.w_layers):
            x = torch.relu(conv(x) + w(x))
        x = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =====================================================
#       3D Volume Super-Resolution (FNO3D)
# =====================================================
def fno3d_super_resolve(volume, input_shape, output_shape, epochs=500, modes=12, width=32):
    """
    volume: 3D numpy array of shape (D, H, W)
    input_shape: shape to downsample to (D, H, W)
    output_shape: target high-resolution shape (D, H, W)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from scipy.ndimage import zoom
    low_res = zoom(volume, zoom=[i / j for i, j in zip(input_shape, volume.shape)], order=1)
    low_res_up = zoom(low_res, zoom=[i / j for i, j in zip(output_shape, low_res.shape)], order=1)
    target_highres = zoom(volume, zoom=[i / j for i, j in zip(output_shape, volume.shape)], order=1)

    input_tensor = torch.tensor(low_res_up, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    target_tensor = torch.tensor(target_highres, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    B, C, D, H, W = input_tensor.shape
    gridx = torch.linspace(0, 1, D)
    gridy = torch.linspace(0, 1, H)
    gridz = torch.linspace(0, 1, W)
    grid = torch.stack(torch.meshgrid(gridx, gridy, gridz, indexing="ij"), dim=-1).unsqueeze(0).to(device)

    input_with_grid = torch.cat([input_tensor.permute(0, 2, 3, 4, 1), grid], dim=-1)

    model = FNO3D(modes, modes, modes, width, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(input_with_grid)
        loss = criterion(output.squeeze(), target_tensor.squeeze())
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        pred = model(input_with_grid).squeeze().cpu().numpy()

    return pred

# =====================================================
#      Load Stack of Grayscale Images as Volume
# =====================================================
def load_grayscale_stack(folder_path, target_size=(64, 64)):
    """
    Load a stack of grayscale images and return a 3D numpy array (D, H, W)
    """
    image_files = sorted([
        f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files:
        raise FileNotFoundError("No image files found in the folder.")

    images = []
    for file in image_files:
        img = Image.open(os.path.join(folder_path, file)).convert("L")
        img = img.resize(target_size, Image.BILINEAR)
        images.append(np.array(img))

    volume = np.stack(images, axis=0)  # shape: (D, H, W)
    return volume

# =====================================================
#                  Example Usage
# =====================================================
if __name__ == "__main__":
    folder_path = "rgb_blob_stack"  # Replace with your folder
    grayscale_volume = load_grayscale_stack(folder_path, target_size=(32, 32))
    grayscale_volume = grayscale_volume.transpose(1, 2, 0)  # (D, H, W) → (H, W, D)
    print("Loaded volume shape:", grayscale_volume.shape)

    super_resolved = fno3d_super_resolve(
        volume=grayscale_volume,
        input_shape=(64, 64, 100),
        output_shape=(256, 256, 100),
        epochs=800
    )

    # Visualization (central slice comparison)
    slice_idx = grayscale_volume.shape[2] // 2
    original_slice = grayscale_volume[:, :, slice_idx]
    super_res_slice = super_resolved[:, :, slice_idx]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_slice, cmap='gray')
    plt.title("Original Slice")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(super_res_slice, cmap='gray')
    plt.title("Super-Resolved Slice")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
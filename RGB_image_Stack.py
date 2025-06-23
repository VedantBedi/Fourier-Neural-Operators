import torch
import torch.nn as nn
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.ndimage import zoom

# ===========================================================
#              Load a 3D Stack of RGB Images
# ===========================================================
def load_rgb_stack(folder_path, target_size=(64, 64)):
    files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    imgs = []
    for fn in files:
        im = Image.open(os.path.join(folder_path, fn)).convert("RGB")
        im = im.resize(target_size, Image.BILINEAR)
        imgs.append(np.array(im))  # shape: (H, W, 3)

    vol = np.stack(imgs, axis=0)             # shape: (D, H, W, 3)
    return np.transpose(vol, (1, 2, 0, 3))   # shape: (H, W, D, 3)

# ===========================================================
#                 3D Spectral Convolution
# ===========================================================
class SpectralConv3d(nn.Module):
    def __init__(self, in_c, out_c, m1, m2, m3):
        super().__init__()
        self.weights = nn.Parameter(
            (1 / (in_c * out_c)) * torch.randn(in_c, out_c, m1, m2, m3, dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_ft = torch.fft.rfftn(x, dim=[2, 3, 4])
        out_ft = torch.zeros_like(x_ft, dtype=torch.cfloat)

        m1, m2, m3 = self.weights.shape[2:]
        out_ft[:, :, :m1, :m2, :m3] = torch.einsum(
            "bixyz,ioxyz->boxyz", x_ft[:, :, :m1, :m2, :m3], self.weights
        )

        return torch.fft.irfftn(out_ft, s=(D, H, W), dim=[2, 3, 4])

# ===========================================================
#                     FNO3D for RGB Volumes
# ===========================================================
class FNO3D(nn.Module):
    def __init__(self, m1, m2, m3, width, in_c=3, out_c=3):
        super().__init__()
        self.fc0 = nn.Linear(in_c + 3, width)  # RGB + coords

        self.convs = nn.ModuleList([
            SpectralConv3d(width, width, m1, m2, m3) for _ in range(4)
        ])
        self.ws = nn.ModuleList([
            nn.Conv3d(width, width, 1) for _ in range(4)
        ])

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Sequential(
            nn.Linear(128, out_c),
            nn.Sigmoid()  # Output in [0,1]
        )

    def forward(self, x):
        x = self.fc0(x)                # (B, H, W, D, 6) -> (B, H, W, D, width)
        x = x.permute(0, 4, 1, 2, 3)   # (B, width, H, W, D)
        for conv, w in zip(self.convs, self.ws):
            x = torch.relu(conv(x) + w(x))
        x = x.permute(0, 2, 3, 4, 1)   # (B, H, W, D, width)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)             # (B, H, W, D, 3)

# ===========================================================
#       3D RGB Super-Resolution with FNO3D + Noise
# ===========================================================
def fno3d_rgb_super_resolve(volume, in_shape, out_shape,
                            epochs=500, modes=8, width=16, noise_std=0.01):
    """
    Args:
        volume: np.array of shape (H, W, D, 3)
        in_shape: target low-res shape
        out_shape: target high-res shape
    Returns:
        np.array of shape (H, W, D, 3) with values clipped in [0, 1]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    volume = volume.astype(np.float32) / 255.0  # Normalize to [0,1]

    def resize(vol, shape):
        out = []
        for c in range(3):
            out.append(zoom(vol[..., c],
                            [shape[0]/vol.shape[0],
                             shape[1]/vol.shape[1],
                             shape[2]/vol.shape[2]], order=1))
        return np.stack(out, axis=-1)

    low = resize(volume, in_shape)
    low_up = resize(low, out_shape)
    tgt = resize(volume, out_shape)

    inp = torch.tensor(low_up, dtype=torch.float32, device=device).unsqueeze(0)
    inp += noise_std * torch.randn_like(inp)

    tgt = torch.tensor(tgt, dtype=torch.float32, device=device).unsqueeze(0)

    B, H, W, D, C = inp.shape

    # Grid coordinates
    gx = torch.linspace(0, 1, H, device=device)
    gy = torch.linspace(0, 1, W, device=device)
    gz = torch.linspace(0, 1, D, device=device)
    grid = torch.stack(torch.meshgrid(gx, gy, gz, indexing="ij"), dim=-1).unsqueeze(0)

    model_input = torch.cat([inp, grid], dim=-1)

    model = FNO3D(modes, modes, modes, width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(model_input).squeeze(0)
        loss = loss_fn(pred, tgt.squeeze(0))
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        out = model(model_input).squeeze(0).cpu().numpy()

    return np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)

# ===========================================================
#                    Example Usage
# ===========================================================
if __name__ == "__main__":
    folder = "rgb_blob_stack"  # Replace with your folder
    vol = load_rgb_stack(folder, target_size=(32, 32))  # shape: (H, W, D, 3)
    print("Volume shape:", vol.shape)

    out = fno3d_rgb_super_resolve(
        vol,
        in_shape=(64, 64, 100),
        out_shape=(256, 256, 100),
        epochs=800,
        modes=16,
        width=32,
        noise_std=0.0
    )

    mid = out.shape[2] // 2  # Central slice in depth
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(vol[:, :, mid, :])
    ax[0].set_title("Original RGB Slice")
    ax[0].axis("off")

    ax[1].imshow(out[:, :, mid, :])
    ax[1].set_title("FNO Super-Resolved Slice")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()
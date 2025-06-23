import os
import cv2
import torch
import torch.nn as nn
import torch.fft
import numpy as np
from scipy.ndimage import zoom
import imageio
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1) Video I/O
# -------------------------------------------------------------------
def load_video_to_volume(video_path, target_size):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
        frames.append(resized)

    cap.release()
    vol = np.stack(frames, axis=0)                  # (D, H, W, 3)
    return np.transpose(vol, (1, 2, 0, 3))           # (H, W, D, 3)


def save_volume_as_video(vol, out_path, fps=30):
    """
    vol: float32 np array in [0,1] with shape (H, W, D, 3)
    """
    writer = imageio.get_writer(
        out_path, format="FFMPEG", mode="I", fps=fps,
        codec="libx264", bitrate="16M"
    )
    for i in range(vol.shape[2]):
        frame = (vol[:, :, i, :] * 255).astype(np.uint8)
        writer.append_data(frame)
    writer.close()


# -------------------------------------------------------------------
# 2) FNO3D Model Definition
# -------------------------------------------------------------------
class SpectralConv3d(nn.Module):
    def __init__(self, in_c, out_c, m1, m2, m3):
        super().__init__()
        self.weights = nn.Parameter((1 / (in_c * out_c)) *
                                    torch.randn(in_c, out_c, m1, m2, m3, dtype=torch.cfloat))

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_ft = torch.fft.rfftn(x, dim=[2, 3, 4])
        out_ft = torch.zeros_like(x_ft, dtype=torch.cfloat)
        m1, m2, m3 = self.weights.shape[2:]
        out_ft[:, :, :m1, :m2, :m3] = torch.einsum(
            "bixyz,ioxyz->boxyz", x_ft[:, :, :m1, :m2, :m3], self.weights)
        return torch.fft.irfftn(out_ft, s=(D, H, W), dim=[2, 3, 4])


class FNO3D(nn.Module):
    def __init__(self, m1, m2, m3, width, in_c=3, out_c=3):
        super().__init__()
        self.fc0 = nn.Linear(in_c + 3, width)
        self.convs = nn.ModuleList([SpectralConv3d(width, width, m1, m2, m3) for _ in range(4)])
        self.ws = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(4)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Sequential(nn.Linear(128, out_c), nn.Sigmoid())

    def forward(self, x):
        x = self.fc0(x)                       # [B, H, W, D, 6]
        x = x.permute(0, 4, 1, 2, 3)          # [B, width, H, W, D]
        for conv, w in zip(self.convs, self.ws):
            x = torch.relu(conv(x) + w(x))
        x = x.permute(0, 2, 3, 4, 1)          # [B, H, W, D, width]
        x = torch.relu(self.fc1(x))
        return self.fc2(x)                    # [B, H, W, D, 3]


# -------------------------------------------------------------------
# 3) Single-clip Super-resolution
# -------------------------------------------------------------------
def process_clip(clip_vol, in_shape, out_shape, modes_x, modes_t, width, epochs, device):
    vol = clip_vol.astype(np.float32) / 255.0

    def resize(v, shape):
        h2, w2, d2 = shape
        return np.stack([
            zoom(v[..., c], [h2 / v.shape[0], w2 / v.shape[1], d2 / v.shape[2]], order=1)
            for c in range(3)
        ], axis=-1)

    low     = resize(vol, in_shape)
    low_up  = resize(low, out_shape)
    target  = resize(vol, out_shape)

    inp = torch.tensor(low_up, dtype=torch.float32, device=device).unsqueeze(0)
    tgt = torch.tensor(target, dtype=torch.float32, device=device).unsqueeze(0)

    B, H, W, D, C = inp.shape
    gx = torch.linspace(0, 1, H, device=device)
    gy = torch.linspace(0, 1, W, device=device)
    gz = torch.linspace(0, 1, D, device=device)
    grid = torch.stack(torch.meshgrid(gx, gy, gz, indexing="ij"), dim=-1).unsqueeze(0)

    model_in = torch.cat([inp, grid], dim=-1)  # [1, H, W, D, 6]

    model = FNO3D(modes_x, modes_x, modes_t, width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(model_in).squeeze(0)
        loss = loss_fn(pred, tgt.squeeze(0))
        loss.backward()
        optimizer.step()
        if ep % 50 == 0:
            print(f"Epoch {ep}, Loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        out = model(model_in).squeeze(0).cpu().numpy()
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)


# -------------------------------------------------------------------
# 4) Main Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Load entire video
    vol = load_video_to_volume("clip.mp4", target_size=(64, 64))
    H, W, D, _ = vol.shape
    print(f"Loaded video: {D} frames of size {H}×{W}")

    # Select 120 random frames
    np.random.seed(0)
    frame_count = min(D, 120)
    idx = np.sort(np.random.choice(D, size=frame_count, replace=False))
    clip = vol[:, :, idx, :]
    print("Random 120-frame clip:", clip.shape)

    # Save input clip (normalized) at 15 FPS
    clip_norm = clip.astype(np.float32) / 255.0
    save_volume_as_video(clip_norm, "input_clip.mp4", fps=15)
    print("Wrote input_clip.mp4")

    # Process using FNO
    sr_vol = process_clip(
        clip,
        in_shape=(128, 128, frame_count),
        out_shape=(256, 256, 180),
        modes_x=15,
        modes_t=15,
        width=32,
        epochs=400,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Super-res volume:", sr_vol.shape)

    # Save output video at 30 FPS
    save_volume_as_video(sr_vol, "output.mp4", fps=30)
    print("Wrote output.mp4")
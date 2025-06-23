import torch
import torch.nn as nn
import torch.fft
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, Resize
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
#                Spectral Convolution Layer
# =====================================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft, dtype=torch.cfloat)

        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x

# =====================================================
#                Fourier Neural Operator 2D
# =====================================================
class FNO2D(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2D, self).__init__()
        self.fc0 = nn.Linear(3, width)

        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # BCHW

        x = torch.relu(self.conv0(x) + self.w0(x))
        x = torch.relu(self.conv1(x) + self.w1(x))
        x = torch.relu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)

        x = x.permute(0, 2, 3, 1)  # BHWC
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =====================================================
#                Padding Helper
# =====================================================
def zero_pad_grayscale(img: np.ndarray, pad_shape=(32, 32)):
    H, W = img.shape
    P_H, P_W = pad_shape
    pad_img = np.zeros((P_H, P_W), dtype=np.uint8)
    start_y = (P_H - H) // 2
    start_x = (P_W - W) // 2
    pad_img[start_y:start_y+H, start_x:start_x+W] = img
    return pad_img

# =====================================================
#           FNO-based Grayscale Super-Resolution
# =====================================================
def fno_super_resolve_grayscale(
    image,
    pad_shape=(32, 32),
    target_resolution=(1024, 1024),
    modes=16,
    width=32,
    epochs=500,
    plot=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert input to numpy and prepare HR target
    original_np = np.array(image)
    H, W = original_np.shape
    target_transform = Resize(target_resolution)
    high_res_image = target_transform(image)

    # Zero pad and convert to tensors
    padded_np = zero_pad_grayscale(original_np, pad_shape)
    padded_image = Image.fromarray(padded_np)

    input_tensor = ToTensor()(padded_image).unsqueeze(0).to(torch.float32).to(device)
    target_tensor = ToTensor()(high_res_image).unsqueeze(0).to(torch.float32).to(device)

    # Add coordinate grid
    B, _, X, Y = input_tensor.shape
    gridx = torch.linspace(0, 1, X)
    gridy = torch.linspace(0, 1, Y)
    grid = torch.stack(torch.meshgrid(gridx, gridy, indexing="ij"), dim=-1).unsqueeze(0).to(device)
    input_with_grid = torch.cat([input_tensor.permute(0, 2, 3, 1), grid], dim=-1)

    # Model setup
    model = FNO2D(modes, modes, width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(input_with_grid)

        output_resized = Resize(target_resolution)(output.permute(0, 3, 1, 2))
        loss = criterion(output_resized.squeeze(), target_tensor.squeeze())

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    # Inference
    model.eval()
    with torch.no_grad():
        pred = model(input_with_grid)
        pred_resized = Resize(target_resolution)(pred.permute(0, 3, 1, 2)).squeeze().cpu().numpy()

    # Visualization
    if plot:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(original_np, cmap="gray")

        plt.subplot(1, 3, 2)
        plt.title("Padded Input")
        plt.imshow(padded_np, cmap="gray")

        plt.subplot(1, 3, 3)
        plt.title(f"FNO Output ({target_resolution[0]}×{target_resolution[1]})")
        plt.imshow(pred_resized, cmap="gray")
        plt.tight_layout()
        plt.show()

    return pred_resized

# =====================================================
#                Entry Point
# =====================================================
if __name__ == "__main__":
    original = Image.open("image.jpg").convert("L")
    resized = original.resize((256, 256))
    output = fno_super_resolve_grayscale(
        image=resized,
        pad_shape=(512, 512),
        target_resolution=(1024, 1024),
        epochs=800
    )
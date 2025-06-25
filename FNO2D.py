import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
# Fourier Neural Operator for 2D data

class SpectralConv2d(nn.Module):
    """
    Spectral Convolutional 2D Layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes1 (int): Number of Fourier modes to keep in x-direction.
        modes2 (int): Number of Fourier modes to keep in y-direction.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to keep in x-direction
        self.modes2 = modes2  # Number of Fourier modes to keep in y-direction

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(
            in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        """
        Complex multiplication for 2D Fourier coefficients.
        
        Args:
            input: (batch, in_channel, x, y)
            weights: (in_channel, out_channel, x, y)
        
        Returns:
            (batch, out_channel, x, y)
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        Forward pass of the 2D spectral convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channel, x, y)

        Returns:
            torch.Tensor: Output tensor in physical space of shape (batch, out_channel, x, y)
        """
        batchsize = x.shape[0]
        size_x = x.shape[-2]
        size_y = x.shape[-1]

        # Perform 2D Fourier transform
        x_ft = torch.fft.rfft2(x, norm='ortho')  # [B, C_in, H, W//2+1]

        # Prepare output tensor in frequency space
        out_ft = torch.zeros(batchsize, self.out_channels, size_x, size_y // 2 + 1,
                             device=x.device, dtype=torch.cfloat)

        # Apply spectral conv to top-left modes
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

        # Transform back to physical space
        x = torch.fft.irfft2(out_ft, s=(size_x, size_y), norm='ortho')  # [B, C_out, H, W]

        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        """
        2D Fourier Neural Operator.

        Args:
            modes1 (int): Number of Fourier modes in the x-direction.
            modes2 (int): Number of Fourier modes in the y-direction.
            width (int): Width of the feature maps (number of channels).
        """
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # Input has 3 channels: [a(x,y), x, y] or [u0(x,y), x, y]
        self.fc0 = nn.Linear(3, self.width)

        # Spectral layers
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # Final projection to scalar output
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape [Batch, Nx, Ny, 3]
        
        Returns:
            Tensor of shape [Batch, Nx, Ny]
        """
        # Lift input to higher dimension
        x = self.fc0(x)                 # [B, Nx, Ny, 3] -> [B, Nx, Ny, width]
        x = x.permute(0, 3, 1, 2)       # [B, width, Nx, Ny]

        # Spectral blocks
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.relu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.relu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.relu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # Project back to output dimension
        x = x.permute(0, 2, 3, 1)       # [B, Nx, Ny, width]
        x = self.fc1(x)                 # [B, Nx, Ny, 128]
        x = F.relu(x)
        x = self.fc2(x)                 # [B, Nx, Ny, 1]
        x = x.squeeze(-1)               # [B, Nx, Ny]

        return x
    




def heat_2d(nu=0.01, N=64, Nt=800, T=1.0, num_samples=1000):
    """
    Generate 2D heat equation data.
    
    Args:
        nu (float): diffusion coefficient
        N (int): grid size (NxN)
        Nt (int): number of time steps
        T (float): total time
        num_samples (int): number of samples

    Returns:
        data: [B, T, N, N] tensor
        x, y: [N] spatial grids
    """
    dx = 1 / (N - 1)
    dy = dx
    dt = T / Nt

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)

    data = np.zeros((num_samples, Nt, N, N), dtype=np.float32)

    for i in range(num_samples):
        # Random smooth initial condition
        X, Y = np.meshgrid(x, y, indexing='ij')
        u = np.sin(2 * np.pi * X * (1 + 0.5 * np.random.rand())) * np.sin(2 * np.pi * Y * (1 + 0.5 * np.random.rand()))
        u *= (0.5 + 0.5 * np.random.rand())

        snapshots = [u.copy()]
        for _ in range(1, Nt):
            u_new = u.copy()
            u_new[1:-1, 1:-1] += nu * dt * (
                (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
                (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
            )
            u = u_new
            snapshots.append(u.copy())

        data[i] = np.stack(snapshots, axis=0)

    return torch.tensor(data, dtype=torch.float32), \
           torch.tensor(x, dtype=torch.float32), \
           torch.tensor(y, dtype=torch.float32)

class Heat2DDataset(Dataset):
    def __init__(self, data, x, y, time_step=-1):
        """
        data: [B, T, N, N]
        x, y: [N] spatial grids
        time_step: which time step to predict (e.g., -1 = final)
        """
        self.data = data
        self.x = x
        self.y = y
        self.time_step = time_step

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        u0 = self.data[idx, 0]                     # [N, N] initial condition
        target = self.data[idx, self.time_step]    # [N, N] target

        # Generate meshgrid: [N, N, 2]
        X, Y = torch.meshgrid(self.x, self.y, indexing='ij')
        input_grid = torch.stack([u0, X, Y], dim=-1)  # [N, N, 3]

        return input_grid, target
    
def train_fno(model, dataloader, epochs=100, lr=0.001, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in dataloader:
            # x: [B, N, N, 3], y: [B, N, N]
            x, y = x.to(device), y.to(device)

            # DO NOT permute here!  FNO2d.forward expects [B, N, N, 3]
            pred = model(x)           # � [B, N, N]
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.6f}")

    return model


def test_resolution_generalization(model, nu=0.01, N=64, Nt=800, T=1.0, time_step=99, device='cuda'):
    model = model.to(device)
    model.eval()

    test_data, x_grid, y_grid = heat_2d(nu=nu, N=N, Nt=Nt, T=T, num_samples=100)
    test_dataset = Heat2DDataset(test_data, x_grid, y_grid, time_step=time_step)
    test_loader  = DataLoader(test_dataset, batch_size=10)

    total_loss = 0.0
    loss_fn     = torch.nn.MSELoss()

    with torch.no_grad():
        for x, y in test_loader:
            # x: [B, N, N, 3], y: [B, N, N]
            x, y = x.to(device), y.to(device)
            # DO NOT permute here either
            pred = model(x)            # � [B, N, N]
            total_loss += loss_fn(pred, y).item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test on Nx={N} | MSE Loss: {avg_loss:.6f}")


def test_time_generalization(model, nu=0.01, N=64, Nt=800, T=1.0, time_step=49, device='cuda'):
    model = model.to(device)
    model.eval()

    test_data, x_grid, y_grid = heat_2d(nu=nu, N=N, Nt=Nt, T=T, num_samples=100)
    test_dataset = Heat2DDataset(test_data, x_grid, y_grid, time_step=time_step)
    test_loader  = DataLoader(test_dataset, batch_size=10)

    total_loss = 0.0
    loss_fn     = torch.nn.MSELoss()

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            # Again, no permute
            pred = model(x)            # � [B, N, N]
            total_loss += loss_fn(pred, y).item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test at t={T * time_step / Nt:.2f} | MSE Loss: {avg_loss:.6f}")

    
if __name__ == "__main__":
    # model = FNO2d(modes1=16,modes2 = 16, width=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nu = 0.01
    data, x, y = heat_2d(nu=nu, N=64, Nt=800, T=1.0, num_samples=1000)
    dataset = Heat2DDataset(data, x, y, time_step=99)
    train_loader = DataLoader(dataset, batch_size=20, shuffle=True)

    model = FNO2d(modes1=16, modes2=16, width=64)
    trained_model = train_fno(model, train_loader, epochs=100, lr=0.001)
    torch.save(trained_model.state_dict(), "fno2d_heat.pth")

    # model = FNO2d(modes1=16, modes2=16, width=64)
    # model.load_state_dict(torch.load("fno2d_heat.pth", map_location=device))
    # model.to(device)
    model.eval()
    # test_resolution_generalization(model, N=32)
    # test_resolution_generalization(model, N=128)
    # test_time_generalization(model, time_step=399)






    nu = 0.01
    N = 64
    Nt = 800
    T = 1

    # Create test data
    test_data, x_grid, y_grid = heat_2d(nu=nu, N=N, Nt=Nt, T=T, num_samples=10)

    # Use time_step=99 as before
    test_dataset = Heat2DDataset(test_data, x_grid, y_grid, time_step=99)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        x_sample, y_true = next(iter(test_loader))
        x_sample, y_true = x_sample.to(device), y_true.to(device)
        y_pred =  model(x_sample)  # [1, N, N]

        # Compute MSE for this sample
        mse_loss_fn = nn.MSELoss()
        mse_value = mse_loss_fn(y_pred, y_true).item()

    # Convert tensors to CPU and numpy for plotting
    u0 = x_sample[0, ..., 0].cpu().numpy()          # initial condition
    ground_truth = y_true[0].cpu().numpy()           # true evolved field
    prediction = y_pred[0].cpu().numpy()             # predicted evolved field
    all_vals = np.concatenate([
        ground_truth.flatten(),
        prediction.flatten()
    ])
    vmax = np.max(np.abs(all_vals))
    vmin = -vmax

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: Initial Condition (we can use the same vmin/vmax so that the three maps line up colorwise)
    im0 = axes[0].imshow(u0,
                        extent=(0, 1, 0, 1),
                        origin='lower',
                        vmin=vmin, vmax=vmax,
                        cmap='viridis')
    axes[0].set_title("Initial Condition")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Middle: Ground Truth (same vmin/vmax)
    im1 = axes[1].imshow(ground_truth,
                        extent=(0, 1, 0, 1),
                        origin='lower',
                        vmin=vmin, vmax=vmax,
                        cmap='viridis')
    axes[1].set_title("Ground Truth")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Right: FNO Prediction (same vmin/vmax)
    im2 = axes[2].imshow(prediction,
                        extent=(0, 1, 0, 1),
                        origin='lower',
                        vmin=vmin, vmax=vmax,
                        cmap='viridis')
    axes[2].set_title("FNO Prediction")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()





import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

# Fourier Neural Operator for 1D data

class SpectralConv1d(nn.Module):
    """
    Spectral (Fourier) Convolution 1D Layer.

    Args:
        in_channels (int): Number of input channels (features).
        out_channels (int): Number of output channels.
        modes (int): Number of low-frequency Fourier modes to keep.
    """
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # number of Fourier modes to keep

        self.scale = 1 / (in_channels * out_channels)
        # Complex weights for the low-frequency modes
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, x_ft, weights):
        # x_ft:  [batch, in_ch, Nx//2+1]
        # weights: [in_ch, out_ch, modes]
        # returns: [batch, out_ch, modes]
        return torch.einsum("bix,iox->box", x_ft, weights)

    def forward(self, x):
        """
        x: [B, in_channels, Nx] (real)
        returns: [B, out_channels, Nx]
        """
        B, C, N = x.shape
        # FFT along the last dimension
        x_ft = torch.fft.rfft(x, norm='ortho')       # [B, C_in, N//2+1] (complex)

        # Allocate output in Fourier domain
        out_ft = torch.zeros(B, self.out_channels, N//2 + 1,
                             device=x.device, dtype=torch.cfloat)

        # Multiply only the lowest 'modes' modes
        out_ft[:, :, :self.modes] = self.compl_mul1d(
            x_ft[:, :, :self.modes], self.weights
        )

        # Inverse FFT back to real space
        x_ifft = torch.fft.irfft(out_ft, n=N, norm='ortho')  # [B, out_ch, N]
        return x_ifft


class FNO1d(nn.Module):
    def __init__(self, modes, width):
        """
        1D Fourier Neural Operator.

        Args:
            modes (int): Number of Fourier modes to keep.
            width (int): Width of the hidden channels.
        """
        super(FNO1d, self).__init__()
        self.modes = modes
        self.width = width

        # Input has 2 channels: [u0(x), x]
        self.fc0 = nn.Linear(2, self.width)

        # Four spectral + pointwise blocks
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        # Final projection to scalar output
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        x: Tensor of shape [B, Nx, 2]  (channels: [u0(x), x-coordinate])
        returns: Tensor of shape [B, Nx]  (the predicted u(x, t+1))
        """
        # Lift input to higher dimension
        x = self.fc0(x)           # [B, Nx, 2] � [B, Nx, width]
        x = x.permute(0, 2, 1)    # [B, width, Nx]

        # Four spectral + pointwise blocks
        x1 = self.conv0(x)        # [B, width, Nx]
        x2 = self.w0(x)           # [B, width, Nx]
        x = F.relu(x1 + x2)

        x1 = self.conv1(x); x2 = self.w1(x); x = F.relu(x1 + x2)
        x1 = self.conv2(x); x2 = self.w2(x); x = F.relu(x1 + x2)
        x1 = self.conv3(x); x2 = self.w3(x); x = x1 + x2

        # Project back to a scalar
        x = x.permute(0, 2, 1)    # [B, Nx, width]
        x = self.fc1(x)           # [B, Nx, 128]
        x = F.relu(x)
        x = self.fc2(x)           # [B, Nx, 1]
        x = x.squeeze(-1)         # [B, Nx]
        return x


# ---------------------------------
# 2) 1D Heat Equation Data (FTCS)
# ---------------------------------

def heat_1d(nu=0.01, N=64, Nt=256, T=1.0, num_samples=1000):
    """
    Generate 1D heatequation (diffusion) data with FTCS (explicit Euler).

    du/dt = nu * u_xx on x[0,1], Dirichlet boundaries u=0 at x=0,1.

    Args:
        nu (float): diffusion coefficient
        N (int): number of spatial points
        Nt (int): number of time steps
        T (float): total time
        num_samples (int): number of random samples to generate

    Returns:
        data: torch.FloatTensor of shape [num_samples, Nt, N]
        x:    torch.FloatTensor of shape [N]
    """
    dx = 1.0 / (N - 1)
    dt = T / Nt

    # CFL check: nu * dt / dx^2 <= 0.25
    cfl = nu * dt / (dx * dx)
    if cfl > 0.25:
        raise ValueError(f"Unstable FTCS: nu�dt/dx� = {cfl:.4f} > 0.25")

    x = np.linspace(0, 1, N, dtype=np.float32)
    data = np.zeros((num_samples, Nt, N), dtype=np.float32)

    for i in range(num_samples):
        # Random smooth initial condition: sine wave with random freq & amplitude
        freq = 1.0 + 0.5 * np.random.rand()
        amp = 0.5 + 0.5 * np.random.rand()
        u = amp * np.sin(2.0 * np.pi * freq * x)

        # Enforce Dirichlet BCs: u[0]=u[-1]=0
        u[0] = 0.0
        u[-1] = 0.0

        snapshots = [u.copy()]
        for n in range(1, Nt):
            u_new = u.copy()
            u_new[1:-1] = u[1:-1] + nu * dt * (
                (u[2:] - 2*u[1:-1] + u[:-2]) / (dx * dx)
            )
            # Reenforce Dirichlet
            u_new[0] = 0.0
            u_new[-1] = 0.0
            u = u_new
            snapshots.append(u.copy())

        data[i] = np.stack(snapshots, axis=0)

    return torch.tensor(data, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)


# -------------------------
# 3) Dataset & DataLoader
# -------------------------

class Heat1DDataset(Dataset):
    def __init__(self, data, x, time_step=-1):
        """
        data: [B, Nt, N]
        x:    [N] spatial grid
        time_step: index of which timeslice to predict (e.g. -1 for final)
        """
        self.data = data
        self.x = x
        self.time_step = time_step

        # Normalize initialand-final pairs (optional but recommended)
        u0 = data[:, 0, :].reshape(data.shape[0], -1)       # [B, N]
        uf = data[:, self.time_step, :].reshape(data.shape[0], -1)  # [B, N]
        all_vals = torch.cat([u0, uf], dim=1)
        self.mu = all_vals.mean().item()
        self.std = all_vals.std().item()
        if self.std < 1e-6:
            self.std = 1.0

        self.data = (self.data - self.mu) / self.std

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        u0 = self.data[idx, 0, :]                   # [N]
        uf = self.data[idx, self.time_step, :]      # [N]
        # Build 2channel input: [u0(x), x]
        xx = self.x                               # [N]
        inp = torch.stack([u0, xx], dim=-1)       # [N, 2]
        return inp, uf


# -------------------------
# 4) Training & Evaluation
# -------------------------

def train_fno_1d(model, dataloader, epochs=100, lr=1e-3, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in dataloader:
            # x_batch: [B, N, 2], y_batch: [B, N]
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)           # [B, N]
            loss = loss_fn(pred, y_batch)   # scalar

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch:3d}/{epochs} | Train MSE = {epoch_loss:.6e}")

    return model


def test_fno_1d(model, nu=0.01, N=64, Nt=256, T=1.0, time_step=-1, device='cuda'):
    model = model.to(device)
    model.eval()

    test_data, x_grid = heat_1d(nu=nu, N=N, Nt=Nt, T=T, num_samples=100)
    test_dataset = Heat1DDataset(test_data, x_grid, time_step=time_step)
    test_loader  = DataLoader(test_dataset, batch_size=10)

    total_loss = 0.0
    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)  # [B, N, 2], [B, N]
            pred = model(x)                    # [B, N]
            total_loss += loss_fn(pred, y).item() * x.size(0)

    avg_loss = total_loss / len(test_loader.dataset)
    print(f"Test at t = {T * (time_step+1) / Nt:.3f} | MSE Loss: {avg_loss:.6e}")


# ----------------------------
# 5) Plotting a Single Sample
# ----------------------------

def plot_single_1d(u0, u_true, u_pred, x):
    """
    Plot the initial condition, ground truth, and FNO prediction on the same axes.

    u0, u_true, u_pred: 1D numpy arrays of shape [N]
    x: 1D numpy array of shape [N]
    """
    plt.figure(figsize=(8, 4))
    plt.plot(x, u0, 'k-', label="Initial (t=0)")
    plt.plot(x, u_true, 'b--', label="Ground Truth (t_final)")
    plt.plot(x, u_pred, 'r-.', label="FNO Prediction")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("1D Diffusion: Initial vs. Ground Truth vs. FNO")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ----------------------------
# 6) Main: Train / Load / Plot
# ----------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # 6a) Generate 1D diffusion data
    # ---------------------------
    nu = 0.01
    N  = 64
    T  = 1.0       # total time we want to be able to predict up to
    # Choose Nt so that nu*dt/dx^2 <= 0.25:
    dx = 1.0/(N-1)
    min_Nt = int(np.ceil((4 * nu * T)/(dx*dx)))  # ensure stability
    Nt = max(min_Nt, 800)  # pick something >= min_Nt
    print(f"Using Nt = {Nt} for stability (nu�dt/dx^2 H {nu*(T/Nt)/(dx*dx):.3f} d 0.25)")

    data, x_grid = heat_1d(nu=nu, N=N, Nt=Nt, T=T, num_samples=1000)
    dataset = Heat1DDataset(data, x_grid, time_step=Nt-1)  # predict final time t=T
    train_loader = DataLoader(dataset, batch_size=20, shuffle=True, drop_last=True)

    # ---------------------------
    # 6b) Build / Train or Load a pretrained FNO1d
    # ---------------------------
    model = FNO1d(modes=16, width=64).to(device)

    # If you already have saved weights, load them; otherwise train from scratch:
    # model.load_state_dict(torch.load("fno1d_heat.pth", map_location=device))
    # print("Loaded pretrained weights.")
    model = train_fno_1d(model, train_loader, epochs=50, lr=1e-3, device=device)
    torch.save(model.state_dict(), "fno1d_heat.pth")

    # ---------------------------
    # 6c) Test resolution and time generalization
    # ---------------------------
    test_fno_1d(model, nu=nu, N=32, Nt=Nt, T=T, time_step=Nt-1, device=device)
    print("\nResolution generalization tests:")
    test_fno_1d(model, nu=nu, N=32, Nt=Nt, T=T, time_step=Nt-1, device=device)
    test_fno_1d(model, nu=nu, N=128, Nt=Nt, T=T, time_step=Nt-1, device=device)

    print("\nTime generalization test (midtime):")
    test_fno_1d(model, nu=nu, N=N, Nt=Nt, T=T, time_step=(Nt//2)-1, device=device)
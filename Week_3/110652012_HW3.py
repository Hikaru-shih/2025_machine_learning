import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Runge function and derivative
def runge(x):
    return 1 / (1 + 25 * x**2)

def runge_derivative(x):
    return -50 * x / (1 + 25 * x**2)**2

# Data
x = np.linspace(-1, 1, 200).reshape(-1, 1)
y = runge(x)
y_prime = runge_derivative(x)

split = int(0.8 * len(x))
x_train, y_train, yprime_train = x[:split], y[:split], y_prime[:split]
x_val, y_val, yprime_val = x[split:], y[split:], y_prime[split:]

x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True)
y_train = torch.tensor(y_train, dtype=torch.float32)
yprime_train = torch.tensor(yprime_train, dtype=torch.float32)

x_val = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
y_val = torch.tensor(y_val, dtype=torch.float32)
yprime_val = torch.tensor(yprime_val, dtype=torch.float32)

# Neural Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
    def forward(self, x):
        return self.layers(x)

model = Net()
mse = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_losses, val_losses = [], []

# Training loop
for epoch in range(1000):
    # Training
    model.train()
    optimizer.zero_grad()

    y_pred = model(x_train)
    dydx = torch.autograd.grad(
        y_pred, x_train,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True
    )[0]

    loss_func = mse(y_pred, y_train)
    loss_deriv = mse(dydx, yprime_train)
    loss = loss_func + loss_deriv

    loss.backward()
    optimizer.step()

    # Validation (⚠ 不要用 no_grad，因為要算導數)
    model.eval()
    y_pred_val = model(x_val)
    dydx_val = torch.autograd.grad(
        y_pred_val, x_val,
        grad_outputs=torch.ones_like(y_pred_val),
        create_graph=True
    )[0]

    val_loss_func = mse(y_pred_val, y_val)
    val_loss_deriv = mse(dydx_val, yprime_val)
    val_loss = val_loss_func + val_loss_deriv

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

# Plot function approximation
x_plot = torch.linspace(-1, 1, 200).reshape(-1,1)
x_plot.requires_grad_(True)
y_plot = model(x_plot)
dydx_plot = torch.autograd.grad(
    y_plot, x_plot,
    grad_outputs=torch.ones_like(y_plot),
    create_graph=False
)[0]

plt.figure()
plt.plot(x, y, label="True Runge f(x)")
plt.plot(x_plot.detach(), y_plot.detach(), label="NN Approximation f(x)")
plt.legend()
plt.savefig("function_vs_nn.png")
plt.close()

plt.figure()
plt.plot(x, y_prime, label="True Runge f'(x)")
plt.plot(x_plot.detach(), dydx_plot.detach(), label="NN Approximation f'(x)")
plt.legend()
plt.savefig("derivative_vs_nn.png")
plt.close()

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.yscale("log")
plt.legend()
plt.savefig("loss_curves.png")
plt.close()

print("Final Validation Loss:", val_losses[-1])

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def runge(x):
    return 1 / (1 + 25 * x**2)

x = np.linspace(-1, 1, 200).reshape(-1, 1)
y = runge(x)

split = int(0.8 * len(x))
x_train, y_train = x[:split], y[:split]
x_val, y_val = x[split:], y[split:]

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_losses, val_losses = [], []
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        val_loss = criterion(val_pred, y_val)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

x_plot = torch.linspace(-1, 1, 200).reshape(-1,1)
with torch.no_grad():
    y_plot = model(x_plot)

plt.figure()
plt.plot(x, y, label="True Runge function")
plt.plot(x_plot, y_plot, label="NN Approximation")
plt.legend()
plt.savefig("function_vs_nn.png")
plt.close()

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.yscale("log")
plt.legend()
plt.savefig("loss_curves.png")
plt.close()

mse = criterion(y_plot, torch.tensor(runge(x_plot.numpy()), dtype=torch.float32))
print("Final MSE:", mse.item())

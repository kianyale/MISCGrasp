import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

# Mock DataLoader with 100 batches per epoch
class MockDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, index):
        return torch.randn(10), torch.tensor(1)

data_loader = torch.utils.data.DataLoader(MockDataset(), batch_size=1)
model = torch.nn.Linear(10, 1)

# Optimizer and Scheduler
optimizer = Adam(model.parameters())
scheduler = OneCycleLR(optimizer, max_lr=2e-4, total_steps=2544, div_factor=25, final_div_factor=20)

# Tracking learning rates
learning_rates = []
learning_rates.append(optimizer.param_groups[0]['lr'])

for epoch in range(2544):
    # Mock training step
    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()
    learning_rates.append(optimizer.param_groups[0]['lr'])

# Plot the learning rate schedule
plt.plot(learning_rates)
plt.title("Learning Rate Schedule with OneCycleLR")
plt.xlabel("Iteration")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.show()

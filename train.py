import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Net

EPOCHS, BATCH = 8, 128
device = "cuda" if torch.cuda.is_available() else "cpu"
norm = transforms.Normalize((0.1307,), (0.3081,))

# Data augmentation on the training set only
train_tf = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(), norm,
])
test_tf = transforms.Compose([transforms.ToTensor(), norm])

train_dl = DataLoader(datasets.MNIST("data", train=True, download=True, transform=train_tf), BATCH, shuffle=True)
test_dl = DataLoader(datasets.MNIST("data", train=False, download=True, transform=test_tf), 1000)

model = Net().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)  # learning-rate scheduling


def evaluate():
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    return 100 * correct / len(test_dl.dataset)


for epoch in range(1, EPOCHS + 1):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    sched.step()
    print(f"epoch {epoch}/{EPOCHS}  test accuracy: {evaluate():.2f}%")

torch.save(model.state_dict(), "model.pt")
print("Saved model.pt")

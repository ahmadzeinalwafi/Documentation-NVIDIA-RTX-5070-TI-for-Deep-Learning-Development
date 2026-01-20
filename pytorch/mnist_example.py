import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    val_ds   = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    class CNN_LSTM_MNIST(nn.Module):
        def __init__(self):
            super().__init__()

            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)   # 28x28 → 14x14
            )

            self.lstm = nn.LSTM(
                input_size=32 * 14,
                hidden_size=128,
                num_layers=2,
                dropout=0.3,
                batch_first=True
            )

            self.fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )

        def forward(self, x):
            x = self.cnn(x)
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.view(x.size(0), 14, -1)
            x, _ = self.lstm(x)
            return self.fc(x[:, -1])

    model = CNN_LSTM_MNIST().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    def run_epoch(loader, train=True):
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.set_grad_enabled(train):
            for data, target in loader:
                data = data.to(device)
                target = target.to(device)

                if train:
                    optimizer.zero_grad()

                output = model(data)
                loss = criterion(output, target)

                if train:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * data.size(0)
                preds = output.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)

        avg_loss = total_loss / total
        acc = correct / total * 100
        return avg_loss, acc

    epochs = 5
    print("\nStarting HARD MNIST training...\n")

    for epoch in range(epochs):
        start = time.time()

        train_loss, train_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader, train=False)

        elapsed = time.time() - start

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"  Time: {elapsed:.1f}s\n")

    print("✅ Training & device test completed")

    if device.type == "cuda":
        print("\nGPU Memory Usage:")
        print("  Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
        print("  Reserved: ", torch.cuda.memory_reserved() / 1024**2, "MB")

if __name__ == "__main__":
    main()

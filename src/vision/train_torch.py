import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from .dataset_cv2 import CIFAR10Cv2Dataset
from .model_torch import SmallCNN

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "outputs/vision"
    os.makedirs(out_dir, exist_ok=True)

    # torchvision returns PIL images; we convert from tensor->PIL if needed
    base_train = CIFAR10(root="data", train=True, download=True, transform=ToPILImage())
    base_test  = CIFAR10(root="data", train=False, download=True, transform=ToPILImage())

    train_ds = CIFAR10Cv2Dataset(base_train, train=True)
    test_ds  = CIFAR10Cv2Dataset(base_test, train=False)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = SmallCNN(num_classes=10).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 6):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in tqdm(train_loader, desc=f"epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_acc = correct / total
        train_loss = total_loss / total

        # quick eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        test_acc = correct / total

        print(f"epoch={epoch} loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

    torch.save(model.state_dict(), f"{out_dir}/smallcnn_cifar10.pt")
    print("Saved:", f"{out_dir}/smallcnn_cifar10.pt")

if __name__ == "__main__":
    main()

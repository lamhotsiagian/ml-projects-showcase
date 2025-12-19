import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class CIFAR10Cv2Dataset(Dataset):
    """
    Wraps torchvision CIFAR-10 but applies OpenCV preprocessing before returning tensors.
    """
    def __init__(self, base_ds, train: bool = True):
        self.base_ds = base_ds
        self.train = train

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img_pil, label = self.base_ds[idx]
        img = np.array(img_pil)  # RGB uint8

        # OpenCV expects BGR often; we'll stay RGB but use cv2 ops safely
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)

        if self.train:
            # simple augmentation: horizontal flip
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)

        # normalize to float32 [0,1]
        img = img.astype(np.float32) / 255.0

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class DeepHandoverDataset(Dataset):
    def __init__(self, session_dir, transform=None, target_transform=None):
        annotations_file = os.path.join('..', 'data', session_dir, session_dir + '.csv')
        self.img_labels = pd.read_csv(annotations_file, sep=' ')
        self.session_dir = session_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.session_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]

        image_tensor = self.transform(image)

        force_tensor = torch.Tensor([
            self.img_labels["fx"],
            self.img_labels["fy"],
            self.img_labels["fz"],
            self.img_labels["mx"],
            self.img_labels["my"],
            self.img_labels["mz"]
        ])

        gripper_state_tensor = torch.Tensor([
            self.img_labels["gripper_is_open"]
        ])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image_tensor, force_tensor, gripper_state_tensor

if __name__ == "__main__":
    data = DeepHandoverDataset("2021-12-01-15:30:36")
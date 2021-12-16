import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class DeepHandoverDataset(Dataset):
    def __init__(self, session_id, transform=None, target_transform=None):
        current_dirname = os.path.dirname(__file__)
        data_dir = os.path.join(current_dirname, '../../data')
        annotations_file = os.path.join(data_dir, session_id, session_id + '.csv')
        self.img_labels = pd.read_csv(annotations_file, sep=' ')
        self.session_id = session_id
        self.transform = transform
        self.target_transform = target_transform

        if self.transform == None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        
        # Load entire dataset into RAM
        self.image_tensors = []
        self.force_tensors = []
        self.gripper_state_tensors = []
        for idx in range(self.img_labels.shape[0]):
            img_path = os.path.join(data_dir, self.session_id, 'images', self.img_labels.iloc[idx, 0])
            image = Image.open(img_path)
            label = self.img_labels.iloc[idx, 1]

            image_tensor = self.transform(image)

            force_tensor = torch.Tensor([
                self.img_labels["fx"][idx],
                self.img_labels["fy"][idx],
                self.img_labels["fz"][idx],
                self.img_labels["mx"][idx],
                self.img_labels["my"][idx],
                self.img_labels["mz"][idx]
            ])

            gripper_state_tensor = torch.Tensor([
                self.img_labels["gripper_is_open"][idx]
            ])

            self.image_tensors.append(image_tensor)
            self.force_tensors.append(force_tensor)
            self.gripper_state_tensors.append(gripper_state_tensor)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return self.image_tensors[idx], self.force_tensors[idx], self.gripper_state_tensors[idx]

if __name__ == "__main__":
    data = DeepHandoverDataset("2021-12-01-15:30:36")
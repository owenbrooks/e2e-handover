import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import os

class DeepHandoverDataset(Dataset):
    def __init__(self, session_id, transform=None, target_transform=None):
        current_dirname = os.path.dirname(__file__)
        data_dir = os.path.join(current_dirname, '../../../data')
        annotations_file = os.path.join(data_dir, session_id, session_id + '.csv')
        self.img_labels = pd.read_csv(annotations_file, sep=' ')
        self.session_id = session_id
        self.transform = transform
        self.target_transform = target_transform
        self.use_segmentation = os.environ.get('use_segmentation')

        if self.transform == None:
            # first three values are standard for ResNet
            mean = [0.485, 0.456, 0.406, 0.5] if self.use_segmentation else [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225, 0.225] if self.use_segmentation else [0.229, 0.224, 0.225]

            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std)
            ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        current_dirname = os.path.dirname(__file__)
        data_dir = os.path.join(current_dirname, '../../../data')
        image_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(data_dir, self.session_id, 'images', image_name)
        image = Image.open(img_path).convert('RGB') # this RGB conversion means it works on the binary segmented images too

        if self.use_segmentation:
            # stack segmented image to create a 4D image
            segmented_image_path = os.path.join(data_dir, self.session_id, 'seg_images', image_name)
            segmented_image = Image.open(segmented_image_path).convert()
            rgb_np = np.asarray(image)
            seg_np = np.asarray(segmented_image)
            image_np = np.stack([rgb_np[:,:,0], rgb_np[:,:,1], rgb_np[:,:,2], seg_np], axis=-1)
            stacked_image = Image.fromarray(image_np)

            image_tensor = self.transform(stacked_image)
        else:
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

        sample = {}
        sample['image'] = image_tensor
        sample['force'] = force_tensor
        sample['gripper_is_open'] = gripper_state_tensor

        return image_tensor, force_tensor, gripper_state_tensor

if __name__ == "__main__":
    data = DeepHandoverDataset("2021-12-01-15:30:36")
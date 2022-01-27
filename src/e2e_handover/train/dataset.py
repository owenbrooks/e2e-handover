#!/usr/bin/env python3
import argparse
from collections import namedtuple
import os
import pandas as pd
from PIL import Image
import sys
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import yaml

class DeepHandoverDataset(Dataset):
    def __init__(self, params, transform=None):
        annotations_file = os.path.join(params.data_file) # path to csv file
        data_dir = os.path.dirname(annotations_file)

        self.img_labels = pd.read_csv(annotations_file, sep=' ')
        self.data_dir = data_dir
        self.main_transform = transform
        self.transform_by_image_type = {}
        self.params = params

        if self.main_transform == None:
            # first three values are standard from ImageNet
            mean_vals = {
                'image_rgb_1': ([0.485, 0.456, 0.406], params.use_rgb_1),
                'image_rgb_2': ([0.485, 0.456, 0.406], params.use_rgb_2),
                'image_depth_1': ([8.8149e-07], params.use_depth_1),
                'image_depth_2': ([1.0863e-06], params.use_depth_2),
                'image_seg_1': ([0.485, 0.456, 0.406], params.use_segmentation and params.use_rgb_1),
                'image_seg_2': ([0.485, 0.456, 0.406], params.use_segmentation and params.use_rgb_2),
            }
            std_vals = {
                'image_rgb_1': [0.229, 0.224, 0.225],
                'image_rgb_2': [0.229, 0.224, 0.225],
                'image_depth_1': [1.6628e-06],
                'image_depth_2': [1.2508e-06],
                'image_seg_1': [0.229, 0.224, 0.225],
                'image_seg_2': [0.229, 0.224, 0.225],
            }
            mean = []
            std = []
            for image_key in mean_vals.keys():
                if mean_vals[image_key][1]:
                    mean = mean_vals[image_key][0]
                    std = std_vals[image_key]

                    self.transform_by_image_type[image_key] = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.ConvertImageDtype(torch.float),
                        transforms.Normalize(mean=mean,std=std)
                    ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        use_images = {
            'image_rgb_1': self.params.use_rgb_1,
            'image_rgb_2': self.params.use_rgb_2,
            'image_depth_1': self.params.use_depth_1,
            'image_depth_2': self.params.use_depth_2,
            'image_seg_1': self.params.use_segmentation,
            'image_seg_2': self.params.use_segmentation,
        }
        image_tensors = []
        for key in use_images.keys():
            if use_images[key]:
                image_rel_path = self.img_labels[key][0]
                img_path = os.path.join(self.data_dir, image_rel_path)
                image = Image.open(img_path).convert()
                if self.main_transform is None:
                    image_t = self.transform_by_image_type[key](image)
                else:
                    image_t = self.main_transform(image)
                image_tensors.append(image_t)

        stacked_image_t = torch.cat(image_tensors, 0) # concatenates into signal tensor with number of channels = sum of channels of each tensor

        force_tensor = torch.Tensor([
            self.img_labels["fx"][idx],
            self.img_labels["fy"][idx],
            self.img_labels["fz"][idx],
            self.img_labels["mx"][idx],
            self.img_labels["my"][idx],
            self.img_labels["mz"][idx]
        ])

        gripper_state_tensor = torch.Tensor([self.img_labels["gripper_is_open"][idx].item()])

        sample = {}
        sample['image'] = stacked_image_t
        sample['force'] = force_tensor
        sample['gripper_is_open'] = gripper_state_tensor

        if self.params.output_velocity:
            vel_cmd_tensor = torch.Tensor([
                self.img_labels["vx"][idx],
                self.img_labels["vy"][idx],
                self.img_labels["vz"][idx],
                self.img_labels["wx"][idx],
                self.img_labels["wy"][idx],
                self.img_labels["wz"][idx],
            ])
            sample['vel_cmd'] = vel_cmd_tensor

        return sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path of csv file to train on e.g. data/2021-12-09-04:56:05/raw.csv')

    current_dirname = os.path.dirname(__file__)
    params_path = os.path.join(current_dirname, 'params.yaml')
    with open(params_path, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

        args = parser.parse_args()
        params['data_file'] = args.data

        params = namedtuple("Params", params.keys())(*params.values())

        dataset = DeepHandoverDataset(params)

        d = dataset[0]
        # print(dataset[0])
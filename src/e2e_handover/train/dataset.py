#!/usr/bin/env python3
import argparse
from collections import namedtuple
from e2e_handover.image_ops import remove_transparency
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
        if params.use_lstm:
            self.sequence_length = params.lstm_sequence_length
        annotations_file = os.path.join(params.data_file) # path to csv file
        data_dir = os.path.dirname(annotations_file)

        self.img_labels = pd.read_csv(annotations_file, sep=' ').iloc[::params.frame_skip, :] # keeps only every nth frame
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
        if self.params.use_lstm:
            image_samples = []
            force_samples = []
            gripper_samples = []
            if idx >= self.sequence_length - 1: # don't need to pad
                # get previous sequence_length rows - 1 rows in addition
                idx_start = idx - self.sequence_length + 1
                idx_end = idx + 1
                for i in range(idx_start, idx_end):
                    sample = self.load_single_item(idx)
                    image_samples.append(sample['image'])
                    force_samples.append(sample['force'])
                    gripper_samples.append(sample['gripper_is_open'])
            else: # pad to sequence_length by repeating first item
                first_sample = self.load_single_item(0)
                samples_to_pad_by = self.sequence_length - idx - 1
                for i in range(samples_to_pad_by):
                    image_samples.append(first_sample['image'])
                    force_samples.append(first_sample['force'])
                    gripper_samples.append(first_sample['gripper_is_open'])
                for i in range(0, idx + 1):
                    sample = self.load_single_item(i)
                    image_samples.append(sample['image'])
                    force_samples.append(sample['force'])
                    gripper_samples.append(sample['gripper_is_open'])
            sequence_sample = {}
            sequence_sample['image'] = torch.stack(image_samples)
            sequence_sample['force'] = torch.stack(force_samples)
            sequence_sample['gripper_is_open'] = torch.stack(gripper_samples)
            return sequence_sample
        else:
            return self.load_single_item(idx)

    def load_single_item(self, idx):

        use_images = {
            'image_rgb_1': self.params.use_rgb_1,
            'image_depth_1': self.params.use_depth_1,
            'image_seg_1': self.params.use_segmentation,
            'image_rgb_2': self.params.use_rgb_2,
            'image_depth_2': self.params.use_depth_2,
            'image_seg_2': self.params.use_segmentation,
        }
        image_tensors = []
        for key in use_images.keys():
            if use_images[key]:
                image_rel_path = self.img_labels[key].iloc[idx]
                # print(image_rel_path)
                img_path = os.path.join(self.data_dir, image_rel_path)
                image = Image.open(img_path).convert()
                image = remove_transparency(image) # removes the alpha channel if present and replaces with black background
                if self.main_transform is None:
                    image_t = self.transform_by_image_type[key](image)
                else:
                    image_t = self.main_transform(image)
                image_tensors.append(image_t)

        stacked_image_t = torch.cat(image_tensors, 0) # concatenates into signal tensor with number of channels = sum of channels of each tensor

        force_tensor = torch.Tensor([
            self.img_labels["fx"].iloc[idx],
            self.img_labels["fy"].iloc[idx],
            self.img_labels["fz"].iloc[idx],
            self.img_labels["mx"].iloc[idx],
            self.img_labels["my"].iloc[idx],
            self.img_labels["mz"].iloc[idx]
        ])

        gripper_state_tensor = torch.Tensor([self.img_labels["gripper_is_open"].iloc[idx].item()])

        sample = {}
        sample['image'] = stacked_image_t
        sample['force'] = force_tensor
        sample['gripper_is_open'] = gripper_state_tensor

        if self.params.output_velocity:
            vel_cmd_tensor = torch.Tensor([
                self.img_labels["vx"].iloc[idx],
                self.img_labels["vy"].iloc[idx],
                self.img_labels["vz"].iloc[idx],
                self.img_labels["wx"].iloc[idx],
                self.img_labels["wy"].iloc[idx],
                self.img_labels["wz"].iloc[idx],
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
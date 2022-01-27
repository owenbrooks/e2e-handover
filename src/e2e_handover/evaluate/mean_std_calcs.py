#!/usr/bin/env python3
import argparse
from collections import namedtuple
from e2e_handover.train.dataset import DeepHandoverDataset
import os
import sys
import torch
from torchvision import transforms
import yaml

# Method as described by zzzf at https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/39
def main(params):
    transform = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float)
    ])

    dataset = DeepHandoverDataset(params, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    channel_count = 0

    channel_addition = { 
        'use_rgb_1': 3,
        'use_rgb_2': 3,
        'use_depth_1': 1,
        'use_depth_2': 1,
    }
    used_images = [key for key in param_dict.keys() if param_dict[key]]
    for key in used_images:
        if key in channel_addition.keys():
            channel_count += channel_addition[key]

    mean = torch.zeros(channel_count)
    std = torch.zeros(channel_count)

    for i, data in enumerate(dataloader):
        # if (i % 10000 == 0): print(i)
        data = data['image'].squeeze(0)
        if (i == 0): size = data.size(1)*data.size(2)
        mean += data.sum((1,2)) / size

    mean /= len(dataloader)
    print(mean)
    mean = mean.unsqueeze(1).unsqueeze(2)

    for i, data in enumerate(dataloader):
        # if (i % 10000 == 0): print(i)
        data = data['image'].squeeze(0)
        std += ((data - mean) ** 2).sum((1, 2)) / size

    std /= len(dataloader)
    std = std.sqrt()
    print(std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path of csv file to train on e.g. data/2021-12-09-04:56:05/raw.csv')

    current_dirname = os.path.dirname(__file__)
    params_path = os.path.join(current_dirname, '../train/params.yaml')
    with open(params_path, 'r') as stream:
        try:
            param_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

        args = parser.parse_args()
        param_dict['data_file'] = args.data

        params = namedtuple("Params", param_dict.keys())(*param_dict.values())
        main(params)

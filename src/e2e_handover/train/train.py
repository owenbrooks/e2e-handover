#!/usr/bin/env python3
import argparse
from collections import namedtuple
from datetime import datetime
from e2e_handover.train.dataset import DeepHandoverDataset
from e2e_handover.train.model_double import MultiViewResNet
from e2e_handover.train.model import ResNet
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import wandb
import yaml

def main(params):
    print("Beginning training. Data: " + params.data_file)
    print(params)
    dataset = DeepHandoverDataset(params)

    # Split between test and train
    train_fraction = 0.8
    train_length = int(len(dataset)*train_fraction)
    test_length = len(dataset) - train_length

    if params.use_lstm: # Perform a time-series split, not random
        train_data = torch.utils.data.Subset(dataset, range(0, train_length))
        test_data = torch.utils.data.Subset(dataset, range(train_length, len(dataset)))
    else:
        train_data, test_data = random_split(dataset, [train_length, test_length], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=params.num_workers, pin_memory=True)

    # Load pre-trained resnet18 model weights
    model = ResNet(params)
    resnet18_url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
    state_dict = torch.hub.load_state_dict_from_url(resnet18_url)
    model.load_partial_state_dict(state_dict)

    # Set device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))
    model.to(device)

    train(model, train_loader, test_loader, device, params)

def train(model, train_loader, test_loader, device, params):
    model.train()

    # Create directory and path for saving model
    current_dirname = os.path.dirname(__file__)
    model_dir = os.path.join(current_dirname, params.model_directory)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    param_string = build_param_string(params)
    model_path = os.path.join(model_dir, f'model_{timestamp}{param_string}.pt')

    BCE = nn.BCELoss()
    MSE = nn.MSELoss()

    # Training loop
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9)
    for epoch in range(params.num_epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            # get the inputs
            input_img = torch.Tensor(data['image']).to(device)
            input_forces = torch.Tensor(data['force']).to(device)
            target_gripstate = torch.Tensor(data['gripper_is_open']).to(device)

            if params.use_lstm:
                target_gripstate = target_gripstate[:, 4, :].unsqueeze(1) # extract final gripstate from sequence

            if params.output_velocity:
                target_vel_cmd = torch.Tensor(data['vel_cmd']).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(input_img, input_forces)
            pred_gripstate = outputs[:, 0].unsqueeze(1)

            if params.output_velocity:
                pred_vel_cmd = outputs[:, 1:7]
                loss = BCE(pred_gripstate, target_gripstate) + MSE(pred_vel_cmd, target_vel_cmd)
            else:
                loss = BCE(pred_gripstate, target_gripstate)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += float(loss.data)

            if i % params.log_step == params.log_step-1:    # print every few mini-batches
                print('Epoch %d. batch loss: %0.5f' %(epoch + 1, loss.data))

        train_loss = running_loss / len(train_loader)
        test_loss, test_accuracy = test(model, test_loader, BCE, MSE, device, params)

        # Log loss in weights and biases
        wandb.log({"train_loss": train_loss, "test_loss": test_loss, 'test_acc': test_accuracy})
        wandb.watch(model)

        print("Train loss: %0.5f, test loss: %0.5f" % (train_loss, test_loss))

        torch.save(model.state_dict(), model_path)

    print('Finished training')

def build_param_string(params):
    param_string = ""
    if params.use_rgb_1:
        param_string += "_rgb1"
    if params.use_rgb_2:
        param_string += "_rgb2"
    if params.use_segmentation:
        param_string += "_seg"
    if params.use_depth_1:
        param_string += "_depth1"
    if params.use_depth_2:
        param_string += "_depth2"
    if params.use_tactile:
        param_string += "_tact"
    if params.use_force_torque:
        param_string += "_force"

    return param_string

def test(model, test_loader, bce, mse, device, params):
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            # get the inputs
            input_img = torch.Tensor(data['image']).to(device)
            input_forces = torch.Tensor(data['force']).to(device)
            target_gripstate = torch.Tensor(data['gripper_is_open']).to(device)

            if params.output_velocity:
                target_vel_cmd = torch.Tensor(data['vel_cmd']).to(device)

            # forward + backward + optimize
            outputs = model(input_img, input_forces)
            pred_gripstate = outputs[:, 0].unsqueeze(1)

            if params.output_velocity:
                pred_vel_cmd = outputs[:, 1:7]
                loss = bce(pred_gripstate, target_gripstate) + mse(pred_vel_cmd, target_vel_cmd)
            else:
                loss = bce(pred_gripstate, target_gripstate)

            output_thresh = pred_gripstate.cpu().data.numpy() > 0.5
            correct = output_thresh == target_gripstate.cpu().data.numpy().astype(bool)
            correct_sum = np.sum(correct)

            running_correct += correct_sum
            running_total += len(output_thresh)

            # print(f"out: {pred_gripstate.cpu().data.numpy()}, targ: {target_gripstate.cpu().data.numpy().astype(bool)}, correct: {running_correct}/{running_total}")

            # print statistics
            running_loss += float(loss.data)

    test_loss = running_loss / len(test_loader)
    test_accuracy = running_correct / float(running_total)

    return test_loss, test_accuracy


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
        wandb.init(project="e2e-handover", entity="owenbrooks")
        wandb.config.update(params)

        params = namedtuple("Params", params.keys())(*params.values())
        main(params)
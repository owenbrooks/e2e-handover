#!/usr/bin/env python3
from dataset import DeepHandoverDataset
import torch
import torch.nn as nn
import model
import torch.optim as optim
import random
# import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split 
import os
import argparse
# import wandb

# wandb.init(project="e2e-handover", entity="owenbrooks")

def main(args):
    session_id = args.session
    print("Beginning training. Session id: " + session_id)
    dataset = DeepHandoverDataset(session_id)
    # random.shuffle(dataset.img_annotation_path_pairs)

    # Split between test and train
    train_fraction = 0.8
    train_length = int(len(dataset)*train_fraction)
    test_length = len(dataset) - train_length
    train_data, test_data = random_split(dataset, [train_length, test_length])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=args.num_workers)

    # Load pre-trained resnet18 model weights
    net = model.ResNet()
    resnet18_url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
    state_dict = torch.hub.load_state_dict_from_url(resnet18_url)
    net.load_partial_state_dict(state_dict)

    # Set device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))
    net.to(device)

    train(net, train_loader, test_loader, device, args)

def train(net, train_loader, test_loader, device, args):
    net.train()

    # Create directory and path for saving model
    current_dirname = os.path.dirname(__file__)
    model_dir = os.path.join(current_dirname, '../../models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, args.session + '.pt')

    criterion = nn.BCELoss()

    # Training loop
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs
            img = torch.autograd.Variable(data[0]).to(device)
            forces = torch.autograd.Variable(data[1]).to(device)
            gripper_is_open = torch.autograd.Variable(data[2]).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(img, forces)
            loss = criterion(outputs, gripper_is_open)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += float(loss.data)

            if i % args.log_step == args.log_step-1:    # print every few mini-batches
                print('Epoch %d. batch loss: %0.5f' %(epoch + 1, loss.data))

        train_loss = running_loss / len(train_loader)
        test_loss, test_accuracy = test(net, test_loader, criterion, device)

        # Log loss in weights and biases
        # wandb.log({"train_loss": train_loss, "test_loss": test_loss})
        # wandb.watch(net)

        print("Train loss: %0.5f, test loss: %0.5f" % (train_loss, test_loss))

        torch.save(net, model_path)

    # log_predictions(net, test_loader, device)
    print('Finished training')

def log_predictions(net, test_loader, device):
    # create a Table with the same columns as above,
    # plus confidence scores for all labels
    columns=["id", "image", "guess", "truth"]
    test_table = wandb.Table(columns=columns)

    # run inference on every image, assuming my_model returns the
    # predicted label, and the ground truth labels are available
    for img_id, data in enumerate(test_loader):
        img = torch.autograd.Variable(data[0]).to(device)
        forces = torch.autograd.Variable(data[1]).to(device)
        gripper_is_open = torch.autograd.Variable(data[2]).to(device)[0][0]

        guess_label = net(img, forces)[0][0]
        test_table.add_data(img_id, wandb.Image(img), \
                            guess_label, gripper_is_open)

    wandb.log({"table_key": test_table})


def test(net,test_loader,criterion, device):
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    
    for i, data in enumerate(test_loader, 0):
        # get the inputs
        img = torch.autograd.Variable(data[0]).to(device)
        forces = torch.autograd.Variable(data[1]).to(device)
        labels = torch.autograd.Variable(data[2]).to(device)

        # forward + backward + optimize
        outputs = net(img,forces)
        loss = criterion(outputs, labels)

        output_thresh = outputs.cpu().data.numpy() > 0.5

        correct = output_thresh == labels.cpu().data.numpy()
        # print(correct)
        correct_sum = np.sum(correct)

        running_correct += correct_sum
        running_total = running_total + len(output_thresh)

        # print statistics
        running_loss += float(loss.data)

    test_loss = running_loss / len(test_loader)
    test_accuracy = running_correct / float(running_total)

    return test_loss, test_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=str, default="2021-12-01-15:30:36", help='session id of data to train on')

    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()
    print(args)
    # wandb.config.update(args)
    main(args)
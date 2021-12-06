#!/usr/bin/env python
from dataset import DeepHandoverDataset
import torch
import torch.nn as nn
import model
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split 
import os

def main():
    session_id = "2021-12-01-15:30:36"
    print("Beginning training. Session id: " + session_id)
    dataset = DeepHandoverDataset(session_id)
    # random.shuffle(dataset.img_annotation_path_pairs)

    # Split between test and train
    train_fraction = 0.8
    train_length = int(len(dataset)*train_fraction)
    test_length = len(dataset) - train_length
    torch.manual_seed(42)
    train_data, test_data = random_split(dataset, [train_length, test_length])

    print(len(dataset),len(train_data),len(test_data))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers = 4)

    # Load pre-trained resnet18 model weights
    net = model.ResNet()
    resnet18_url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
    state_dict = torch.hub.load_state_dict_from_url(resnet18_url)
    net.load_partial_state_dict(state_dict)

    # Set device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))
    net.to(device)

    train(net, train_loader, test_loader, device)

def train(net, train_loader, test_loader, device):
    train_loss_list = []
    test_loss_list = []
    test_acc_list = []

    net.train()

    # Create directory and path for saving model
    current_dirname = os.path.dirname(__file__)
    model_dir = os.path.join(current_dirname, '../models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(current_dirname, '../models/handover.pt')

    criterion = nn.BCELoss()

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(100):  # loop over the dataset multiple times
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

            print("output", outputs)
            print("loss", loss)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += float(loss.data)

            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] batch loss: %0.5f' %(epoch + 1, i + 1, loss.data))

        train_loss = running_loss / i
        test_loss, test_accuracy = test(net, test_loader, criterion, device)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_accuracy)

        plt.cla()
        plt.plot(train_loss_list, label="train loss")
        plt.plot(test_loss_list, label="test loss")
        plt.plot(test_acc_list, label="test accuracy")
        plt.draw()
        plt.pause(0.1)

        print("Train/Test %0.5f / %0.5f" % (train_loss, test_loss))

        torch.save(net, model_path)
    print('Finished Training')
    plt.show()

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

    test_loss = running_loss / i
    test_accuracy = running_correct / float(running_total)
    print(running_correct,running_total)

    return test_loss, test_accuracy


if __name__ == "__main__":
    main()
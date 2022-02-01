#!/usr/bin/env python3
# Model based on https://github.com/acosgun/deep_handover/blob/master/src/pytorch/model.py
import torch
import torch.nn as nn
from e2e_handover.train.model import ResNet

class ResNetBackbone(ResNet):
    """ Consists of ResNet up to but not including the fully connected layers """
    def forward(self, img):
        if self.params.use_lstm:
            batch_size = img.shape[0]
            seq_length = img.shape[1]
            img = img.reshape([-1, img.shape[2], img.shape[3], img.shape[4]])
            x = self.conv_1(img)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.conv2(x)
            x = self.relu(x)
            x = x.view(batch_size, seq_length, -1)
            return x
        else:
            x = self.bn0(img)
            x = self.conv_1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.conv2(x)
            x = self.relu(x)
            x = self.bn2(x)
            x = x.view(x.size(0), -1)
            return x


class MultiViewResNet(nn.Module):
    """ This model consists of one or optionally two ResNet backbones, 
    followed by fully connected layers of ResNet or optionally an LSTM.
    Options should be configured in params.yaml. It accepts RGB or RGBD 
    images as input. """
    def __init__(self, params):
        super(MultiViewResNet, self).__init__()
        self.params = params
        self.backbone1 = ResNetBackbone(params)
        self.backbone2 = ResNetBackbone(params)

        output_neurons = 7 if params.output_velocity else 1
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        number_of_images = 2 if self.params.use_rgb_1 and self.params.use_rgb_2 else 1
        if self.params.use_lstm:
            self.lstm1 = nn.LSTM(input_size=16*7*7*number_of_images+6, hidden_size=256, batch_first=True)
            self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, output_neurons)
        else:
            self.fc1 = nn.Linear(16*7*7*number_of_images+6, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, output_neurons)

    def forward(self, img, forces):
        concat_dimension = 2 if self.params.use_lstm else 1

        if self.params.use_rgb_1 and self.params.use_rgb_2:
            # split the image tensor into the two images
            channels_per_image = img.shape[1]//2
            img_1 = img[:, :channels_per_image, :, :]
            img_2 = img[:, channels_per_image:, :, :]

            x1 = self.backbone1(img_1)
            x2 = self.backbone2(img_2)
            
            x = torch.cat((x1, x2, forces), dim=concat_dimension)
        else: # only one image
            x = self.backbone1(img)
            x = torch.cat((x, forces), dim=concat_dimension)

        if self.params.use_lstm:
            x, _ = self.lstm1(x)
            x = self.relu(x)
            x, _ = self.lstm2(x)
            x = self.relu(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
        else:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.fc4(x)
            x = self.sigmoid(x)
        return x

    def load_partial_state_dict(self,pretrained_dict):
        """This will only load weights for operations that exist in the model
        and in the state_dict"""

        state_dict = self.state_dict()
        state_dict_b1 = self.backbone1.state_dict()
        state_dict_b2 = self.backbone2.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in state_dict}
        pretrained_dict_b = {k: v for k, v in pretrained_dict.items() if k in state_dict_b1}
        # 2. overwrite entries in the existing state dict
        state_dict.update(pretrained_dict)
        state_dict_b1.update(pretrained_dict_b)
        state_dict_b2.update(pretrained_dict_b)
        # 3. load the new state dict
        self.load_state_dict(state_dict)
        self.backbone1.load_state_dict(state_dict_b1)
        self.backbone2.load_state_dict(state_dict_b2)


if __name__ == "__main__":
    net = MultiViewResNet()
    state_dict = torch.load('resnet18-5c106cde.pth')
    net.load_partial_state_dict(state_dict)

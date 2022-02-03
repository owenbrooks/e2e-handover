import argparse
import cv2
from collections import namedtuple
from e2e_handover.train import model
from e2e_handover.train.dataset import DeepHandoverDataset
from e2e_handover.image_ops import prepare_image
from e2e_handover.segmentation import Segmentor
import numpy as np
import os
import sys
import torch
from torchvision import transforms
import yaml

class NoneTransform(object):
    def __call__(self, image):
        return image

def main(model_name, should_segment, inference_params):
    print(inference_params)

    # Convert to dict so we can edit it so that we can see all views
    viewing_params = dict(inference_params._asdict())
    viewing_params['use_rgb_1'] = True
    viewing_params['use_rgb_2'] = True
    viewing_params['use_depth_1'] = True
    viewing_params['use_depth_2'] = True
    viewing_params = namedtuple("Params", viewing_params.keys())(*viewing_params.values())

    # Load dataset
    viewing_dataset = DeepHandoverDataset(viewing_params, transform=transforms.ToTensor())
    inference_dataset = DeepHandoverDataset(inference_params)

    if should_segment:
        segmentor = Segmentor()

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Create network and load weights
    if not args.no_inference:
        net = model.ResNet(inference_params)
        current_dirname = os.path.dirname(__file__)
        model_path = os.path.join(current_dirname, inference_params.model_directory, model_name)
        net.load_state_dict(torch.load(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device: " + str(device))
        net.to(device)
        net.eval()

    index = 0
    while True:
        sample = viewing_dataset[index]

        images = {}
        images['image_rgb_1'] = sample['image'][0:3, :, :]
        images['image_depth_1'] = sample['image'][3:4, :, :]
        images['image_rgb_2'] = sample['image'][4:7, :, :]
        images['image_depth_2'] = sample['image'][7:8, :, :]

        # Convert single-channels depth images to 3 channels
        images['image_depth_1'] = torch.cat([images['image_depth_1'], images['image_depth_1'], images['image_depth_1']])
        images['image_depth_2'] = torch.cat([images['image_depth_2'], images['image_depth_2'], images['image_depth_2']])

        rgb_images = np.concatenate((images['image_rgb_1'].numpy()[:, ::-1, :], images['image_rgb_2']), axis=2).transpose(1, 2, 0)
        depth_images = np.concatenate((images['image_depth_1'].numpy()[:, ::-1, :], images['image_depth_2']), axis=2).transpose(1, 2, 0)
        img = np.concatenate((rgb_images, depth_images), axis=0)[:, :, ::-1].copy()

        if should_segment:
            binary_mask = segmentor.person_binary_inference(img)
            img = np.array(binary_mask*255, dtype=np.uint8)
            # img = img[binary_mask]
        else:
            image_number_string = str(index) + '/' + str(len(viewing_dataset))
            cv2.putText(img, image_number_string, (0, 460), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

            ground_truth_state = 'open' if sample['gripper_is_open'] else 'closed'
            cv2.putText(img, ground_truth_state, (550, 420), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

            if not args.no_inference:
                inference_sample = inference_dataset[index]
                img_t = inference_sample['image'].unsqueeze(0).to(device)
                forces_t = torch.Tensor(inference_sample['force'].unsqueeze(0)).to(device)

                # forward + backward + optimize
                output_t = net(img_t, forces_t)

                model_output = output_t.cpu().detach().numpy()[0][0]
                model_open = model_output > 0.5

                cv2.putText(img, str(model_output), (500, 50), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                model_state = 'open' if model_open else 'closed'
                cv2.putText(img, model_state, (550, 460), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('Inference ', img)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            index = (index + 1) % len(viewing_dataset)
        elif key == ord('a'):
            index = (index - 1) % len(viewing_dataset)
        elif key == ord('j'):
            index = (index - 100) % len(viewing_dataset)
        elif key == ord('k'):
            index = (index + 100) % len(viewing_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-inference', action='store_true')
    parser.add_argument('--model', type=str, default='2021-12-14-23_calib.pt')
    parser.add_argument('--segment', action='store_true')
    parser.add_argument('--data', type=str, help='path of csv file to run on e.g. data/2021-12-09-04:56:05/raw.csv')

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
        main(args.model, args.segment, params)
import argparse
import cv2
from collections import namedtuple
from e2e_handover.train.model_double import MultiViewResNet
from e2e_handover.train.dataset import DeepHandoverDataset
from e2e_handover.image_ops import prepare_image
# from e2e_handover.segmentation import Segmentor
from rembg import detect
import rembg
import numpy as np
import os
import sys
import torch
from torchvision import transforms
import yaml
from PIL import Image

def main(model_path, should_segment, inference_params):
    print(inference_params)

    # Convert to dict so we can edit it so that we can see all views
    viewing_params = dict(inference_params._asdict())
    viewing_params['use_rgb_1'] = True
    # viewing_params['use_rgb_2'] = True
    viewing_params['use_depth_1'] = True
    # viewing_params['use_depth_2'] = True
    viewing_params['use_lstm'] = False
    viewing_params = namedtuple("Params", viewing_params.keys())(*viewing_params.values())

    # Load dataset
    sample_height = 240
    sample_width = 300*2
    resize = transforms.Compose([ transforms.Resize((sample_height,sample_width//2)), transforms.ToTensor()])
    viewing_dataset = DeepHandoverDataset(viewing_params, transform=resize)
    inference_dataset = DeepHandoverDataset(inference_params)

    ort_session = detect.ort_session('u2netp')

    font = cv2.FONT_HERSHEY_SIMPLEX

    index = 0
    print("background is white, infinity/no return is blue")
    while True:
        sample = viewing_dataset[index]

        images = {}
        images['image_rgb_2'] = sample['image'][0:3, :, :]
        images['image_depth_2'] = sample['image'][3:4, :, :]
        # images['image_rgb_1'] = sample['image'][4:7, :, :]
        # images['image_depth_1'] = sample['image'][7:8, :, :]

        # Convert single-channels depth images to 3 channels
        # images['image_depth_1'] = torch.cat([images['image_depth_1'], images['image_depth_1'], images['image_depth_1']])
        images['image_depth_2'] = torch.cat([images['image_depth_2'], images['image_depth_2'], images['image_depth_2']])

        # Flips camera 2 as it is easier to see image upside down
        # rgb_images = np.concatenate((images['image_rgb_1'], images['image_rgb_2'].numpy()[:, ::-1, :]), axis=2).transpose(1, 2, 0)
        # depth_images = np.concatenate((images['image_depth_1'], images['image_depth_2'].numpy()[:, ::-1, :]), axis=2).transpose(1, 2, 0)
        rgb_images = (images['image_rgb_2'].numpy()[:, :, :]).transpose(1, 2, 0)
        # prediction_img = detect.predict(ort_session, rgb_images)
        # prediction_img.resize((rgb_images.shape[0], rgb_images.shape[1]), Image.ANTIALIAS)
        # bg_mask = np.array(prediction_img) == 0
        bg_mask = rembg.remove((rgb_images*255).astype(np.uint8), only_mask=True, session=ort_session)
        print(rgb_images.shape, bg_mask.shape)
        orig_rgb = rgb_images.copy()
        depth_images = (images['image_depth_2'].numpy()[:, :, :]).transpose(1, 2, 0)
        depth_images /= 65535.0 # for display using opencv
        bg_threshold = 0.02
        # bg_mask = depth_images[:, :, 0] > bg_threshold
        inf_mask = depth_images[:, :, 0] == 0.0

        # checks top row
        inf_columns = depth_images[0, :, 0] == 0.0
        bg_columns = depth_images[0, :, 0] > bg_threshold

        # bg_mask = np.zeros((rgb_images.shape[0], rgb_images.shape[1]), dtype=np.bool)
        # bg_mask[:, bg_columns] = True
        # inf_mask = np.zeros((rgb_images.shape[0], rgb_images.shape[1]), dtype=np.bool)
        # inf_mask[:, inf_columns] = True


        depth_images *= 2.0 # exagerate for display
        inf_mask = denoise(inf_mask)
        # bg_mask = denoise(bg_mask)
        # orig_rgb[inf_mask] = (0, 255, 255)
        # depth_images[inf_mask] = 1.0
        # rgb_images[bg_mask] = (255, 255, 255)
        rgb_images = cv2.bitwise_and(rgb_images, rgb_images, mask=bg_mask)
        # img = np.concatenate((rgb_images, depth_images), axis=0)[:, :, ::-1].copy()
        img = np.concatenate((rgb_images, orig_rgb), axis=0)[:, :, ::-1].copy()

        image_number_string = str(index) + '/' + str(len(viewing_dataset))
        cv2.putText(img, image_number_string, (0, 460), font, 0.8, (100, 10, 90), 1, cv2.LINE_AA)

        ground_truth_state = 'open' if sample['gripper_is_open'] else 'closed'
        cv2.putText(img, ground_truth_state, (550, 420), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Inference', img)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            index = (index + 1) % len(viewing_dataset)
        elif key == ord('a'):
            index = (index - 1) % len(viewing_dataset)
        elif key == ord('j'):
            index = (index - 25) % len(viewing_dataset)
        elif key == ord('k'):
            index = (index + 25) % len(viewing_dataset)
    
def denoise(img):
    img = img*1.0
    kernel = np.ones((5,5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=5)
    img = cv2.erode(img, kernel, iterations=15)
    # img = cv2.dilate(img, kernel, iterations=2)
    # img = cv2.erode(img, kernel, iterations=1)

    return img == 1.0

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
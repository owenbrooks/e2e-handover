import argparse
import cv2
from collections import namedtuple
from e2e_handover.train import model
from e2e_handover.image_ops import prepare_image
from e2e_handover.segmentation import Segmentor
import numpy as np
import os
import pandas as pd
import sys
import torch
import yaml

def main(annotations_path, model_name, should_segment, params):
    data_dir = os.path.dirname(annotations_path)
    df = pd.read_csv(annotations_path, sep=' ')

    if should_segment:
        segmentor = Segmentor()

    font = cv2.FONT_HERSHEY_SIMPLEX

    if not args.no_inference:
        # Create network and load weights
        net = model.ResNet(params)
        current_dirname = os.path.dirname(__file__)
        model_path = os.path.join(current_dirname, '../../../models', model_name)
        net.load_state_dict(torch.load(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device: " + str(device))
        net.to(device)
        net.eval()

    index = 5000
    while True:
        row = df.iloc[index]
        image_rel_path = row['image_rgb_1']
        image_path = os.path.join(data_dir, image_rel_path)
        img = cv2.imread(image_path)

        images = {}
        for key in ['image_rgb_1', 'image_rgb_2', 'image_depth_1', 'image_depth_2']:
            image_rel_path = row[key]
            image_path = os.path.join(data_dir, image_rel_path)
            images[key] = cv2.imread(image_path)
        rgb_images = np.concatenate((images['image_rgb_1'], images['image_rgb_2']), axis=1)
        depth_images = np.concatenate((images['image_depth_1'], images['image_depth_2']), axis=1)

        img = np.concatenate((rgb_images, depth_images), axis=0)

        if should_segment:
            binary_mask = segmentor.person_binary_inference(img)
            img = np.array(binary_mask*255, dtype=np.uint8)
            # img = img[binary_mask]
        else:
            image_number_string = str(index) + '/' + str(len(df.index))
            cv2.putText(img, image_number_string, (0, 460), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

            ground_truth_state = 'open' if row['gripper_is_open'] else 'closed'
            cv2.putText(img, ground_truth_state, (550, 420), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

            if not args.no_inference:
                img_t = prepare_image(img).unsqueeze_(0).to(device)
                wrench_array = row[['fx', 'fy', 'fz', 'mx', 'my', 'mz']].values.astype(np.float32)
                forces_t = torch.Tensor(wrench_array).unsqueeze_(0).to(device)

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
            index = (index + 1) % len(df)
        elif key == ord('a'):
            index = (index - 1) % len(df)
        elif key == ord('j'):
            index = (index - 100) % len(df)
        elif key == ord('k'):
            index = (index + 100) % len(df)

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
        main(args.data, args.model, args.segment, params)
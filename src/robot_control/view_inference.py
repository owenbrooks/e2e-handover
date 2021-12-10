import cv2
import numpy as np
import pandas as pd
import os
import argparse
from robot_control import model
from robot_control.image_ops import prepare_image
import torch

def main(args):
    session_id = '2021-12-09-04:56:05'
    current_dirname = os.path.dirname(__file__)
    data_dir = os.path.join(current_dirname, '../../data')
    annotations_file = os.path.join(data_dir, session_id, session_id + '.csv')
    df = pd.read_csv(annotations_file, sep=' ')

    font = cv2.FONT_HERSHEY_SIMPLEX

    if not args.ground_truth:
        # Create network and load weights
        model_name = session_id + '.pt'
        net = model.ResNet()
        current_dirname = os.path.dirname(__file__)
        model_path = os.path.join(current_dirname, '../../models', model_name)
        net.load_state_dict(torch.load(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device: " + str(device))
        net.to(device)

    index = 0
    while True:
        row = df.iloc[index]
        image_path = os.path.join(data_dir, session_id, row['image_id'])
        img = cv2.imread(image_path)

        image_number_string = str(index) + '/' + str(len(df.index))
        cv2.putText(img, image_number_string, (0, 460), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)


        img_t = prepare_image(img).unsqueeze_(0).to(device)
        wrench_array = row[['fx', 'fy', 'fz', 'mx', 'my', 'mz']].values.astype(np.float32)
        forces_t = torch.autograd.Variable(torch.FloatTensor(wrench_array)).unsqueeze_(0).to(device)

        # forward + backward + optimize
        output_t = net(img_t, forces_t)
        model_output = output_t.cpu().detach().numpy()[0][0]
        model_open = model_output > 0.5

        cv2.putText(img, str(model_output), (500, 50), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        ground_truth_state = 'open' if row['gripper_is_open'] else 'closed'
        model_state = 'open' if model_open else 'closed'
        cv2.putText(img, ground_truth_state, (550, 420), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, model_state, (550, 460), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('Inference ', img)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            index = (index + 1) % len(df)
        elif key == ord('a'):
            index = (index - 1) % len(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground-truth', action='store_true')
    args = parser.parse_args()
    main(args)
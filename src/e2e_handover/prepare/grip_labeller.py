import argparse
import cv2
from collections import namedtuple
from enum import IntEnum

from e2e_handover.train.dataset import DeepHandoverDataset
import numpy as np
import os
import pandas as pd
from torchvision import transforms

class GripperLabel(IntEnum):
    Unchanged=0
    EditedOpen=1
    EditedClosed=2

    def __str__(self):
        if self.value == GripperLabel.Unchanged.value:
            return "unchanged"
        elif self.value == GripperLabel.EditedOpen.value:
            return "open"
        else:
            return "closed"

def main(data_file, use_two_cams):
    params =  {
        'data_file': data_file,
        'use_rgb_1': True,
        'use_rgb_2': use_two_cams,
        'use_depth_1': False,
        'use_depth_2': False,
        'use_segmentation': False,
        'output_velocity': False,
        'use_lstm': False,
        'frame_skip': 1,
    }
    params = namedtuple("Params", params.keys())(*params.values())

    # Load dataset
    viewing_dataset = DeepHandoverDataset(params, transform=transforms.ToTensor())

    font = cv2.FONT_HERSHEY_SIMPLEX

    manual_open_labels = np.zeros(len(viewing_dataset)) # stores GripperLabels
    current_gripper_mode = GripperLabel.Unchanged

    image_height, image_width = 480, 640
    if use_two_cams:
        image_width *= 2
    
    index = 0
    while True:
        sample = viewing_dataset[index]

        image_rgb_1 = sample['image']

        if use_two_cams:
            image_rgb_2 = sample['image'][3:6, :, :].numpy()[:, ::-1, ::-1] # Flip camera 2 as it is easier to see image upside down
            img = np.concatenate((image_rgb_1, image_rgb_2), axis=2).transpose(1, 2, 0)[:, :, ::-1].copy()
        else:
            img = image_rgb_1.numpy().transpose(1, 2, 0)[:, :, ::-1].copy()

        image_number_string = str(index) + '/' + str(len(viewing_dataset))
        cv2.putText(img, image_number_string, (0, 460), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        ground_truth_state = 'open' if sample['gripper_is_open'] else 'closed'
        cv2.putText(img, ground_truth_state, (550, 420), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        # Record label
        if current_gripper_mode != GripperLabel.Unchanged:
            manual_open_labels[index] = current_gripper_mode

        # Display label on image by coloured rectangle
        if current_gripper_mode == GripperLabel.EditedOpen:
            colour = (0, 255, 0)
            cv2.rectangle(img, (0, 0), (image_width, image_height), colour, 3)
        elif current_gripper_mode == GripperLabel.EditedClosed:
            colour = (0, 0, 255)
            cv2.rectangle(img, (0, 0), (image_width, image_height), colour, 3)

        cv2.putText(img, str(GripperLabel(manual_open_labels[index])), (500, 460), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('Annotator', img)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            print("Would you like to save annotations? (y/n)")
            x = input()
            if x.lower()[0] == 'y':
                save_labels(manual_open_labels, data_file)
                print('Saved')
            else:
                print('Did not save')
            break
        # Movement controls
        if key == ord('d'): # go to next frame
            index = (index + 1) % len(viewing_dataset)
        elif key == ord('a'): # go to prev frame
            index = (index - 1) % len(viewing_dataset)
        elif key == ord('j'): # skip ahead a few frames
            index = (index - 25) % len(viewing_dataset)
        elif key == ord('k'): # go back a few frames
            index = (index + 25) % len(viewing_dataset)
        # Gripper Label controls
        elif key == ord('o'): # label open
            current_gripper_mode = GripperLabel.EditedOpen
        elif key == ord('c'): # label closed
            current_gripper_mode = GripperLabel.EditedClosed
        elif key == ord('b'): # no manual label
            current_gripper_mode = GripperLabel.Unchanged
        elif key == ord('m'): # write annotations to disk
            save_labels(manual_open_labels, data_file)

def save_labels(manual_open_labels, data_file):
    data = pd.read_csv(data_file, sep=' ')
    new_labels = data['gripper_is_open']

    manually_opened = manual_open_labels == GripperLabel.EditedOpen
    manually_closed = manual_open_labels == GripperLabel.EditedClosed
    new_labels[manually_opened] = True
    new_labels[manually_closed] = False

    data['gripper_is_open'] = new_labels

    orig_filename = os.path.splitext(os.path.split(data_file)[-1])[-2]
    labelled_path = os.path.join(os.path.dirname(data_file), f'{orig_filename}_labelled.csv')

    data.to_csv(labelled_path, sep=' ', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path of csv file to run on e.g. data/2021-12-09-04:56:05/raw.csv')
    parser.add_argument('--double-camera', action='store_true', help='use two camera views')
    args = parser.parse_args()

    main(args.data, args.double_camera)
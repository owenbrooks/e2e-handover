import argparse
from selectors import EpollSelector
import cv2
from collections import namedtuple
from enum import IntEnum
from e2e_handover.train.dataset import DeepHandoverDataset
import numpy as np
import os
import pandas as pd
from torchvision import transforms

class HandoverType(IntEnum):
    Giving=0
    Receiving=1

    def __str__(self):
        if self.value == HandoverType.Giving:
            return "giving"
        else:
            return "receiving"

def main(data_file):
    params =  {
        'data_file': data_file,
        'use_rgb_1': True,
        'use_rgb_2': False,
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

    actions = np.zeros(len(viewing_dataset)) # stores HandoverType
    current_action = HandoverType.Receiving

    image_height, image_width = 480, 640
    
    index = 0
    while True:
        sample = viewing_dataset[index]

        img = sample['image'].numpy().transpose(1, 2, 0)[:, ::-1, ::-1].copy() # Flip camera 2 as it is easier to see image upside down

        image_number_string = str(index) + '/' + str(len(viewing_dataset))
        cv2.putText(img, image_number_string, (0, 460), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        ground_truth_state = 'open' if sample['gripper_is_open'] else 'closed'
        cv2.putText(img, ground_truth_state, (550, 420), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        # Record label
        actions[index] = current_action

        # Display label on image by coloured rectangle
        if current_action == HandoverType.Receiving:
            colour = (255, 255, 0) # blue for receiving
            cv2.rectangle(img, (0, 0), (image_width, image_height), colour, 3)
        elif current_action == HandoverType.Giving:
            colour = (0, 255, 255) # yellow for giving
            cv2.rectangle(img, (0, 0), (image_width, image_height), colour, 3)

        cv2.putText(img, str(HandoverType(actions[index])), (500, 460), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('Annotator', img)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            print("Would you like to save annotations? (y/n)")
            x = input()
            if x.lower()[0] == 'y':
                save_labels(actions, data_file)
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
            current_action = HandoverType.Giving
        elif key == ord('c'): # label closed
            current_action = HandoverType.Receiving
        elif key == ord('m'): # write annotations to disk
            save_labels(actions, data_file)
        elif key == 32: # spacebar to switch modes
            if current_action == HandoverType.Giving:
                current_action = HandoverType.Receiving
            else:
                current_action = HandoverType.Giving


def save_labels(actions, data_file):
    orig_data = pd.read_csv(data_file, sep=' ')

    receiving_indices = actions == HandoverType.Receiving
    giving_indices = actions == HandoverType.Giving
    
    receiving_data = orig_data[receiving_indices]
    giving_data = orig_data[giving_indices]
    
    orig_filename = os.path.splitext(os.path.split(data_file)[-1])[-2]
    receiving_path = os.path.join(os.path.dirname(data_file), f'{orig_filename}_receive.csv')
    giving_path = os.path.join(os.path.dirname(data_file), f'{orig_filename}_give.csv')

    receiving_data.to_csv(receiving_path, sep=' ', index=False)
    giving_data.to_csv(giving_path, sep=' ', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path of csv file to run on e.g. data/2021-12-09-04:56:05/raw.csv')
    args = parser.parse_args()

    main(args.data)
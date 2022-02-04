
import argparse
import cv2
from collections import namedtuple
from enum import IntEnum
from e2e_handover.train.dataset import DeepHandoverDataset
import numpy as np
import os
import pandas as pd
from torchvision import transforms

class HandoverSwitch(IntEnum):
    GivingToReceiving=0
    ReceivingToGiving=1

    def __str__(self):
        if self.value == HandoverSwitch.GivingToReceiving.value:
            return "giving->receiving"
        else:
            return "receiving->giving"

def find_transition_indices(data_file):
    # read in csv directly to avoid looping through all images from disk
    df = pd.read_csv(data_file, sep=' ')
    indices = np.argwhere(np.diff(df['gripper_is_open'])).squeeze(1)
    return indices

def save_annotations(annotations, data_file):
    data_dir = os.path.dirname(data_file)
    new_filename = os.path.join(data_dir, 'direction.csv')
    out_df = pd.DataFrame(annotations, columns=['index', 'transition_val', 'transition_str'])
    out_df.to_csv(new_filename, sep=' ', index=False)

def main(data_file):
    params =  {
        'data_file': data_file,
        'use_rgb_1': False,
        'use_rgb_2': True,
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

    transition_indices = find_transition_indices(params.data_file)

    annotation_mode = HandoverSwitch.GivingToReceiving
    annotations = []
    
    index = 0
    while True:
        sample = viewing_dataset[index]

        img = sample['image'].numpy().transpose(1, 2, 0)[:, :, ::-1].copy()

        image_number_string = str(index) + '/' + str(len(viewing_dataset))
        cv2.putText(img, image_number_string, (0, 460), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        ground_truth_state = 'open' if sample['gripper_is_open'] else 'closed'
        cv2.putText(img, ground_truth_state, (550, 420), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(img, str(annotation_mode), (400, 460), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('Annotator', img)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            save_annotations(annotations, data_file)
            break
        elif key == ord('d'): # go to next frame
            index = (index + 1) % len(viewing_dataset)
        elif key == ord('a'): # go to prev frame
            index = (index - 1) % len(viewing_dataset)
        elif key == ord('j'): # skip ahead a few frames
            index = (index - 50) % len(viewing_dataset)
        elif key == ord('k'): # go back a few frames
            index = (index + 50) % len(viewing_dataset)
        elif key == ord('l'): # skip ahead to next transition
            later_transitions = transition_indices[transition_indices > index]
            index = later_transitions.min() if len(later_transitions) > 0 else transition_indices[0]
        elif key == ord('h'): # skip back to prev transition
            earlier_transitions = transition_indices[transition_indices < index]
            index = earlier_transitions.max() if len(earlier_transitions) > 0 else transition_indices[-1]
        elif key == ord('t'): # switch annotation mode
            if annotation_mode == HandoverSwitch.GivingToReceiving:
                annotation_mode = HandoverSwitch.ReceivingToGiving
            else:
                annotation_mode = HandoverSwitch.GivingToReceiving
        elif key == 32: # spacebar to add annotation
            if annotation_mode == HandoverSwitch.GivingToReceiving:
                annotation_mode = HandoverSwitch.ReceivingToGiving
            else:
                annotation_mode = HandoverSwitch.GivingToReceiving
            annotations.append((index, int(annotation_mode), annotation_mode))
            print(index, int(annotation_mode), annotation_mode)
        elif key == ord('m'): # write annotations to disk
            save_annotations(annotations, data_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path of csv file to run on e.g. data/2021-12-09-04:56:05/raw.csv')
    args = parser.parse_args()
    main(args.data)
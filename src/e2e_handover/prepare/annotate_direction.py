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

def save_annotations(annotations, data_file, dataset_length):
    starting_transition = 0 if annotations[0][1] == 1 else 1
    start = (0, starting_transition)
    ending_transition = 0 if annotations[-1][1] == 1 else 1
    end = (dataset_length-1, ending_transition)
    
    annotations.append(start)
    annotations.append(end)

    data_dir = os.path.dirname(data_file)
    new_filename = os.path.join(data_dir, 'direction.csv')
    out_df = pd.DataFrame(annotations, columns=['index', 'transition_val', 'transition_str'])
    out_df.to_csv(new_filename, sep=' ', index=False)

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

    transition_indices = find_transition_indices(params.data_file) # points where the gripper changes state

    annotation_mode = HandoverSwitch.ReceivingToGiving
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
            cv2.destroyAllWindows()
            print("Would you like to save annotations? (y/n)")
            x = input()
            if x.lower()[0] == 'y':
                save_annotations(annotations, data_file, len(viewing_dataset))
                print('Saved')
            else:
                print('Did not save')
            break
        if key == ord('d'): # go to next frame
            index = (index + 1) % len(viewing_dataset)
        elif key == ord('a'): # go to prev frame
            index = (index - 1) % len(viewing_dataset)
        elif key == ord('j'): # skip ahead a few frames
            index = (index - 25) % len(viewing_dataset)
        elif key == ord('k'): # go back a few frames
            index = (index + 25) % len(viewing_dataset)
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
            annotations.append((index, int(annotation_mode), str(annotation_mode)))
            print(index, int(annotation_mode), annotation_mode)
            if annotation_mode == HandoverSwitch.GivingToReceiving:
                annotation_mode = HandoverSwitch.ReceivingToGiving
            else:
                annotation_mode = HandoverSwitch.GivingToReceiving
        elif key == ord('m'): # write annotations to disk
            save_annotations(annotations, data_file, len(viewing_dataset))

def apply(data_file):
    orig_data = pd.read_csv(data_file, sep=' ')
    annotations_path = os.path.join(os.path.dirname(data_file), 'direction.csv')
    annotations = pd.read_csv(annotations_path, sep=' ')

    annotations = annotations.sort_values('index')
    annotations.reset_index(drop=True, inplace=True)

    transitions = annotations['transition_val']

    no_double_transitions = np.all(np.diff(transitions).astype(bool))

    if not no_double_transitions:
        raise ValueError(f"Please check the transitions")

    receiving_slices = np.zeros(len(orig_data), dtype=np.bool)
    giving_slices = np.zeros(len(orig_data), dtype=np.bool)

    for i in range(len(annotations)-1):
        start_index = annotations['index'].iloc[i]
        end_index = annotations['index'].iloc[i+1]
        if annotations['transition_val'].iloc[i] == 1: # receiving->giving
            giving_slices[start_index:end_index] = True
        else:
            receiving_slices[start_index:end_index] = True
        print(f'{start_index} {end_index}')

    receiving_data = orig_data[receiving_slices]
    giving_data = orig_data[giving_slices]

    orig_filename = os.path.splitext(os.path.split(data_file)[-1])[-2]
    receiving_file = os.path.join(os.path.dirname(data_file), f'{orig_filename}_receive.csv')
    giving_file = os.path.join(os.path.dirname(data_file), f'{orig_filename}_give.csv')

    receiving_data.to_csv(receiving_file, sep=' ', index=False)
    giving_data.to_csv(giving_file, sep=' ', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path of csv file to run on e.g. data/2021-12-09-04:56:05/raw.csv')
    parser.add_argument('--apply', action='store_true', help='separate the data into giving and receiving CSVs based on direction.csv')
    args = parser.parse_args()

    if args.apply:
        apply(args.data)
    else:
        main(args.data)
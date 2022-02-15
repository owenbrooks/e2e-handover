# Manipulate recorded data
# Instead of gripper_state being binary for open (True) and close (False),
# It will store one of three values: do_nothing (0), trigger_open (1), trigger_close (2)

import argparse
from enum import IntEnum
import os
import pandas as pd
import cv2
from e2e_handover.segmentation import Segmentor
import numpy as np

class GripperAction(IntEnum):
    DoNothing=0
    TriggerClose=1
    TriggerOpen=2

def transition_count(session_id: str):
    current_dirname = os.path.dirname(__file__)
    data_dir = os.path.join(current_dirname, '../../../data')
    annotations_file = os.path.join(data_dir, session_id, session_id + '.csv')
    df = pd.read_csv(annotations_file, sep=' ')

    action_list = []
    for i in range(len(df)-1):
        curr_frame_open = df.iloc[i]['gripper_is_open']
        next_frame_open = df.iloc[i+1]['gripper_is_open']
        if curr_frame_open and not next_frame_open:
            action = GripperAction.TriggerClose
        elif not curr_frame_open and next_frame_open:
            action = GripperAction.TriggerOpen
        else:
            action = GripperAction.DoNothing
        action_list.append(int(action))

    df = pd.Series(action_list)
    print(df.value_counts())

def segment(session_id: str):
    """ Performs segmentation to convert images to a binary mask of person/non-person """
    current_dirname = os.path.dirname(__file__)
    data_dir = os.path.join(current_dirname, '../../../data')
    annotations_file = os.path.join(data_dir, session_id, session_id + '.csv')
    df = pd.read_csv(annotations_file, sep=' ')

    # Create a new folder to store the binary images with background subtracted
    orig_image_dir = os.path.join(data_dir, session_id, 'images',)
    new_image_dir = os.path.join(data_dir, session_id, 'seg_images')
    if not os.path.exists(new_image_dir):
        os.makedirs(new_image_dir)

    segmentor = Segmentor()

    for i in range(len(df)):
        orig_image_path = os.path.join(orig_image_dir, df.iloc[i]['image_id'])
        img = cv2.imread(orig_image_path)

        foreground_mask = segmentor.person_binary_inference(img)
        new_image = np.array(foreground_mask*255, dtype=np.uint8)
        new_image_path = os.path.join(new_image_dir, df.iloc[i]['image_id'])

        cv2.imwrite(new_image_path, new_image)

        print(f"{i+1}/{len(df)} completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=str)
    parser.add_argument('--tcount', action='store_true')
    parser.add_argument('--segment', action='store_true')

    args = parser.parse_args()

    if args.tcount:
        transition_count(args.session)

    if args.segment:
        segment(args.session)

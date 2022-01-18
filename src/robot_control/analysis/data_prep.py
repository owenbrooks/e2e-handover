# Manipulate recorded data
# Instead of gripper_state being binary for open (True) and close (False),
# It will store one of three values: do_nothing (0), trigger_open (1), trigger_close (2)

import argparse
from enum import IntEnum
import os
import pandas as pd
import shutil
import cv2
from robot_control.segmentation import Segmentor
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

def calibrate_forces(session_id: str, static_index: int):
    """ Subtracts the force-torque value at static_index from all the rest to serve as calibration 
    since the force readings seem to change over the day, potentially when the robot is 
    rebooted. """
    current_dirname = os.path.dirname(__file__)
    data_dir = os.path.join(current_dirname, '../../../data')
    annotations_file = os.path.join(data_dir, session_id, session_id + '.csv')
    df = pd.read_csv(annotations_file, sep=' ')
    
    baseline_force = df.iloc[static_index][["fx", "fy", "fz", "mx", "my", "mz"]]
    df[["fx", "fy", "fz", "mx", "my", "mz"]] -= baseline_force

    calib_annotations_file = os.path.join(data_dir, session_id, session_id + '_calib' + '.csv')
    df.to_csv(calib_annotations_file, sep=' ', index=False)
    print(f"Calibrated force readings for {calib_annotations_file}")

def combine_sessions(session_list, out_session_id):
    current_dirname = os.path.dirname(__file__)
    data_dir = os.path.join(current_dirname, '../../../data')
    out_session_dir = os.path.join(data_dir, out_session_id)
    out_annotations_path = os.path.join(out_session_dir, out_session_id + '.csv')

    out_image_dir = os.path.join(out_session_dir, 'images')
    if not os.path.exists(out_session_dir):
        os.makedirs(out_image_dir)

    # Combine csv files
    in_frames = [pd.read_csv(os.path.join(data_dir, session_id, session_id + '.csv'), sep=' ') for session_id in session_list]
    out_df = pd.concat(in_frames)        
    out_df.to_csv(out_annotations_path, sep=' ', index=False)

    print(f'CSV files combined, copying {len(out_df.index)} images (may take a while)')

    # Copy images
    in_image_dirs = [os.path.join(data_dir, session_id, 'images') for session_id in session_list]
    for in_dir in in_image_dirs:
        shutil.copytree(in_dir, out_image_dir, dirs_exist_ok=True)

def segment(session_id: str):
    """ Performs segmentation to convert images to a binary mask of person/non-person """
    current_dirname = os.path.dirname(__file__)
    data_dir = os.path.join(current_dirname, '../../../data')
    annotations_file = os.path.join(data_dir, session_id, session_id + '.csv')
    df = pd.read_csv(annotations_file, sep=' ')

    # Create a new folder to store the binary images with background subtracted
    orig_image_dir = os.path.join(data_dir, session_id, 'images',)
    new_image_dir = os.path.join(data_dir, session_id + '_seg', 'images')
    if not os.path.exists(new_image_dir):
        os.makedirs(new_image_dir)

    # Copy csv file with annotations to the new folder
    csv_copy_path = os.path.join(data_dir, session_id + '_seg', session_id + '_seg.csv')
    df.to_csv(csv_copy_path, sep=' ', index=False)

    segmentor = Segmentor()

    for i in range(len(df)):
        orig_image_path = os.path.join(orig_image_dir, df.iloc[i]['image_id'])
        img = cv2.imread(orig_image_path)

        foreground_mask = segmentor.person_binary_inference(img)
        new_image = np.array(foreground_mask*255, dtype=np.uint8)
        new_image_path = os.path.join(new_image_dir, df.iloc[i]['image_id'])

        cv2.imwrite(new_image_path, new_image)
        # cv2.imshow('mask', foreground_mask)
        # cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=str)
    parser.add_argument('--static-index', default=0, type=int)
    parser.add_argument('--tcount', action='store_true')
    parser.add_argument('--calibf', action='store_true')
    parser.add_argument('--segment', action='store_true')

    subparsers = parser.add_subparsers(dest='subcommand')
    parser_combine = subparsers.add_parser('combine')
    parser_combine.add_argument('-l', nargs="+", required=True, help="list of session ids to combine", dest="session_list")

    args = parser.parse_args()

    if args.tcount:
        transition_count(args.session)

    if args.calibf:
        calibrate_forces(args.session, args.static_index)

    if args.segment:
        segment(args.session)

    if args.subcommand == 'combine':
        combine_sessions(args.session_list, args.session)

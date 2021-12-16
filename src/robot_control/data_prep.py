# Manipulate recorded data
# Instead of gripper_state being binary for open (True) and close (False),
# It will store one of three values: do_nothing (0), trigger_open (1), trigger_close (2)

import argparse
from enum import IntEnum
import os
import pandas as pd

class GripperAction(IntEnum):
    DoNothing=0
    TriggerClose=1
    TriggerOpen=2

def transition_count(session_id: str):
    current_dirname = os.path.dirname(__file__)
    data_dir = os.path.join(current_dirname, '../../data')
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
    data_dir = os.path.join(current_dirname, '../../data')
    annotations_file = os.path.join(data_dir, session_id, session_id + '.csv')
    df = pd.read_csv(annotations_file, sep=' ')
    
    baseline_force = df.iloc[static_index][["fx", "fy", "fz", "mx", "my", "mz"]]
    df[["fx", "fy", "fz", "mx", "my", "mz"]] -= baseline_force

    calib_annotations_file = os.path.join(data_dir, session_id, session_id + '_calib' + '.csv')
    df.to_csv(calib_annotations_file, sep=' ', index=False)
    print(f"Calibrated force readings for {calib_annotations_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=str)
    parser.add_argument('--static-index', default=0, type=int)
    parser.add_argument('--tcount', action='store_true')
    parser.add_argument('--calibf', action='store_true')
    args = parser.parse_args()

    if args.tcount:
        transition_count(args.session)

    if args.calibf:
        calibrate_forces(args.session, args.static_index)
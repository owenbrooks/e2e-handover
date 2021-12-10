# Functions for manipulation recorded data
import argparse
from enum import IntEnum
import os
import pandas as pd
import numpy as np

class GripperAction(IntEnum):
    DoNothing=0
    TriggerOpen=1
    TriggerClose=2

def trigger_only(session_id):
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
    # print(action_list)

    df = pd.Series(action_list)
    print(df.value_counts())

def add_recent_transition(session_id: str, lookback_frames: int):
    # For each frame, look back at the state of the previous recent frames. 
    # If the state changed from open to closed in that time, it is too soon to open again
    # If it changed from closed to open, it is too soon to close again

    # closed, no recent transition
    # closed, recent transition
    # open, no recent transition
    # open, recent transition

    current_dirname = os.path.dirname(__file__)
    data_dir = os.path.join(current_dirname, '../../data')
    annotations_file = os.path.join(data_dir, session_id, session_id + '.csv')
    df = pd.read_csv(annotations_file, sep=' ')

    # Figure out transition indices 
    gripper_is_open = df['gripper_is_open'].values
    transition_indices = np.argwhere(np.diff(gripper_is_open)).squeeze()+1
    recent_transitions = np.zeros(len(gripper_is_open), dtype=bool)

    for trans_index in transition_indices:
        if trans_index > lookback_frames:
            recent_transitions[trans_index-lookback_frames:trans_index] = True
    
    df['recent_transition'] = pd.Series(recent_transitions)

    new_annotations_file = os.path.join(data_dir, session_id, session_id + '_transition.csv')
    df.to_csv(new_annotations_file, sep=' ', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', default='2021-12-09-04:56:05', type=str)
    args = parser.parse_args()
    # trigger_only(args.session)
    add_recent_transition(args.session, 40)
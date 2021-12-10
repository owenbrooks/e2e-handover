# Manipulate recorded data
# Instead of gripper_state being binary for open (True) and close (False),
# It will store one of three values: do_nothing (0), trigger_open (1), trigger_close (2)

import argparse
from enum import IntEnum
import os
import pandas as pd

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

def add_intent(session_id: str, lookback_frames: int):
    pass        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', default='2021-12-09-04:56:05', type=str)
    args = parser.parse_args()
    # trigger_only(args.session)
    add_intent(args.session, 40)
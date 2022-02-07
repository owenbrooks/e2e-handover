
import argparse
import numpy as np
import os
import pandas as pd

def main(data_file, offset, is_giving: bool):
    # Receiving:
    # For each frame, compares against the frame 5 in the future and if it is closed, considers the current frame closed
    # Giving similar but if it is open, considers open
    data = pd.read_csv(data_file, sep=' ')

    gripper_is_open_orig = data['gripper_is_open'].values
    shifted = np.roll(gripper_is_open_orig, -offset)

    if is_giving:
        shifted[-offset:] = False
        future_open = np.logical_or(gripper_is_open_orig, shifted)
        data['gripper_is_open'] = future_open
    else:
        shifted[-offset:] = True
        future_open = np.logical_and(gripper_is_open_orig, shifted)
        data['gripper_is_open'] = future_open

    handover_type = 'giving' if is_giving else 'receiving'
    new_path = os.path.join(os.path.dirname(data_file), f'{handover_type}_offset_{offset}.csv')
    data.to_csv(new_path, sep=' ', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path of csv file to run on e.g. data/2021-12-09-04:56:05/raw.csv')
    parser.add_argument('--offset', type=int)
    parser.add_argument('--giving', action='store_true')
    args = parser.parse_args()

    main(args.data, args.offset, args.giving)

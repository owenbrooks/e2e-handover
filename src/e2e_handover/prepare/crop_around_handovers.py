
import argparse
import numpy as np
import os
import pandas as pd
from numpy.lib.stride_tricks import as_strided

def generate_crop_indices(window_centres, window_size):
    # Crops windows of size window, centred around the open/close transition points
    # See https://stackoverflow.com/a/34191654
    indices = np.arange(np.max(window_centres)+window_size//2+1) # will be int64 -> stride=(8,8)
    s_indices = as_strided(indices, shape=(len(indices)-window_size+1,window_size), strides=(8,8))
    cropped_indices = s_indices[window_centres-window_size//2]
    cropped_indices = cropped_indices.flatten()
    return cropped_indices

def main(data_file, window):
    orig_data = pd.read_csv(data_file, sep=' ')
    print(len(orig_data))
    transitions = np.argwhere(np.diff(orig_data['gripper_is_open'])).squeeze(1)+1

    large_enough = transitions >= window//2
    small_enough = transitions <= len(orig_data) - window//2
    full_window_fits = np.logical_and(large_enough, small_enough)
    transitions_inside = transitions[full_window_fits]
    transitions_below = transitions[np.logical_not(large_enough)]
    transitions_above = transitions[np.logical_not(small_enough)]
    print(f"transitions: {transitions}")
    print(f"transitions fitting inside full window: {transitions_inside}")
    print(f"transitions too early: {transitions_below}")
    print(f"transitions too late: {transitions_above}")

    crops_below = np.arange(np.max(transitions_below)+1) if len(transitions_below) > 0 else np.array([])
    crops_above = np.arange(np.min(transitions_above)+1, len(orig_data)) if len(transitions_above) else np.array([])

    cropped_indices = generate_crop_indices(transitions_inside, window)
    cropped_indices = np.insert(cropped_indices, 0, crops_below)
    cropped_indices = np.append(cropped_indices, crops_above)
    cropped_indices = np.unique(cropped_indices)

    cropped_data = orig_data.iloc[cropped_indices]

    print(f"Reduced number of frames from {len(orig_data)} to {len(cropped_data)}.")

    cropped_file = os.path.join(os.path.dirname(data_file), 'cropped.csv')
    cropped_data.to_csv(cropped_file, sep=' ', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path of csv file to run on e.g. data/2021-12-09-04:56:05/raw.csv')
    parser.add_argument('--window', type=int)
    args = parser.parse_args()

    main(args.data, args.window)
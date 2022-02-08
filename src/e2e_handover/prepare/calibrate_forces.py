import argparse
import os
import pandas as pd

def calibrate_forces(annotations_file: str, static_index: int):
    """ Subtracts the force-torque value at static_index from all the rest to serve as calibration 
    since the force readings seem to change over the day, potentially when the robot is 
    rebooted. """
    df = pd.read_csv(annotations_file, sep=' ')
    
    baseline_force = df.iloc[static_index][["fx", "fy", "fz", "mx", "my", "mz"]]
    df[["fx", "fy", "fz", "mx", "my", "mz"]] -= baseline_force

    orig_filename = os.path.splitext(os.path.split(annotations_file)[-1])[-2]
    calib_annotations_file = os.path.join(os.path.dirname(annotations_file), f'{orig_filename}_calib.csv')
    df.to_csv(calib_annotations_file, sep=' ', index=False)

    print(f"Calibrated force readings for {calib_annotations_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path of csv file to run on e.g. data/2021-12-09-04:56:05/raw.csv')
    args = parser.parse_args()
    calibrate_forces(args.data, 0)
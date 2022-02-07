# Manipulate recorded data
# Instead of gripper_state being binary for open (True) and close (False),
# It will store one of three values: do_nothing (0), trigger_open (1), trigger_close (2)

import argparse
import pandas as pd

def main(data_file):
    df = pd.read_csv(data_file, sep=' ')
    print(df['gripper_is_open'].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)

    args = parser.parse_args()

    main(args.data)
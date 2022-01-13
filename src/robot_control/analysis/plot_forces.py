import cv2
import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt


def main(session_id):
    current_dirname = os.path.dirname(__file__)
    data_dir = os.path.join(current_dirname, '../../../data')
    annotations_file = os.path.join(data_dir, session_id, session_id + '.csv')
    df = pd.read_csv(annotations_file, sep=' ')

    headers = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']
    df[headers].plot()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=str, default='2021-12-14-23_calib_subtracted')
    args = parser.parse_args()
    main(args.session)
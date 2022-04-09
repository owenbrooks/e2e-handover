""" Accepts a dataset csv path as input. Removes the background from each of the images
in the set, saving new images in new respective folders and a new csv. 
e.g. python3 remove_background.py --data data/2022-03-27-04\:16\:35.csv
Goes through image_rgb_1 and image_rgb_1, creates image_rgb_1_rembg and image_rgb_2_rembg 
folders full of images without background.
"""
import argparse
from e2e_handover.bg_segmentation import Segmentor
import pandas as pd

def main(data_csv_path):
    data = pd.read_csv(data_csv_path, sep=' ')
    segmentor = Segmentor()
    image_cols = ['image_rgb_1', 'image_rgb_2']
    actual_image_cols = [col for col in image_cols if col in data.columns]
    image_paths = data[actual_image_cols]
    image_paths_flat = pd.concat([image_paths[col] for col in image_paths], ignore_index=True)

    # create necessary directories to store new images if needed

def save_rembg(orig_image_path: str):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=str, help='path of csv file to run on e.g. data/2021-12-09-04:56:05/raw.csv')
    args = parser.parse_args()

    main(args.data)

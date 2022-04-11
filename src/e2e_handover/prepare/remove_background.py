""" Accepts a dataset csv path as input. Removes the background from each of the images
in the set, saving new images in new respective folders and a new csv. 
e.g. python3 remove_background.py --data data/2022-03-27-04\:16\:35.csv
Goes through image_rgb_1 and image_rgb_1, creates image_rgb_1_rembg and image_rgb_2_rembg 
folders full of images without background.
"""
import argparse
from datetime import datetime
from e2e_handover.bg_segmentation import Segmentor
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
from PIL import Image
import cv2

class BackgroundRemover():
    def __init__(self, data_csv_path):
        data = pd.read_csv(data_csv_path, sep=' ')
        csv_directory = os.path.dirname(data_csv_path)
        image_cols = ['image_rgb_1', 'image_rgb_2']
        actual_image_cols = [col for col in image_cols if col in data.columns]
        image_paths = data[actual_image_cols]
        image_paths_flat = pd.concat([image_paths[col] for col in image_paths], ignore_index=True)

        # create directories to store new images if needed
        split_paths = [os.path.split(path) for path in image_paths_flat.values]
        suffixed_heads = [split[0] + '_rembg' for split in split_paths] # add _rembg suffix to each of the image directories
        unique_directories = np.unique(np.array(suffixed_heads))
        full_directory_paths = [os.path.join(csv_directory, d) for d in unique_directories]

        for folder in full_directory_paths:
            os.makedirs(folder, exist_ok=True)
        print("Created new image directories")

        full_input_paths = [os.path.join(csv_directory, p) for p in image_paths_flat]
        
        self.segmentor = Segmentor()

        # p = Pool()
        # p.map(self.save_rembg, full_input_paths)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"Started at {current_time}.")

        length = len(full_input_paths)
        for i, path in enumerate(full_input_paths):
            self.save_rembg(path)
            print(f"{i+1}/{length} images done.")

    def save_rembg(self, orig_image_path: str):
        # load and transform image
        image = Image.open(orig_image_path).convert()
        # remove background
        result = self.segmentor.inference(np.array(image))

        # save image
        orig_head, tail = os.path.split(orig_image_path)
        new_head = orig_head + '_rembg'
        output_image_path = os.path.join(new_head, tail)
        cv2.imwrite(output_image_path, result[:, :, ::-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=str, help='path of csv file to run on e.g. data/2021-12-09-04:56:05/raw.csv')
    args = parser.parse_args()

    bg_rem = BackgroundRemover(args.data)
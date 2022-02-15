import argparse
import os
import pandas as pd

def combine_files(csv_list: str, out: str):
    """ Subtracts the force-torque value at static_index from all the rest to serve as calibration 
    since the force readings seem to change over the day, potentially when the robot is 
    rebooted. """
    in_frames = []
    for annotations_file in csv_list:
        df = pd.read_csv(annotations_file, sep=' ')
        image_paths = df[['image_rgb_1', 'image_rgb_2', 'image_depth_1', 'image_depth_2']]
        containing_folder = os.path.split(os.path.split(annotations_file)[0])[1]
        new_paths = containing_folder + '/' + image_paths.values
        df[['image_rgb_1', 'image_rgb_2', 'image_depth_1', 'image_depth_2']] = new_paths
        in_frames.append(df)

    # Combine and save csv files
    out_df = pd.concat(in_frames)        
    out_annotations_path = os.path.abspath(out) # path to csv file
    out_dir = os.path.dirname(out_annotations_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_df.to_csv(out_annotations_path, sep=' ', index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs="+", required=True, help="list of csv files to combine", dest="data")
    parser.add_argument('--out', type=str, help="name of output directory", required=True)
    args = parser.parse_args()
    print(args.data)
    combine_files(args.data, args.out)
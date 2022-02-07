import argparse
from collections import namedtuple
from datetime import datetime
from e2e_handover.train.dataset import DeepHandoverDataset
from e2e_handover.train.model import MultiViewResNet
import json
import numpy as np
import os
import torch
from torch.utils.data import random_split 
import yaml

""" This file performs inference using a given model on a dataset and calculates
    statistics giving an idea of the model's accuracy. """

def main(session_id, model_name, test_fraction):
    # Create network and load weights
    net = MultiViewResNet()
    current_dirname = os.path.dirname(__file__)
    model_path = os.path.join(current_dirname, '../../../models', model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load(model_path, map_location=device))
    print("Using device: " + str(device))
    net.to(device)
    net.eval()

    # Load dataset
    dataset = DeepHandoverDataset(session_id)
    train_length = int(len(dataset)*(1.0-test_fraction))
    test_length = len(dataset) - train_length
    _, test_data = random_split(dataset, [train_length, test_length], generator=torch.Generator().manual_seed(42))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    ground_truth_open = np.zeros(test_length, dtype=bool)
    model_open = np.zeros_like(ground_truth_open)

    # Perform inference on all test images
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            img = torch.Tensor(data[0]).to(device)
            forces = torch.Tensor(data[1]).to(device)
            ground_truth = torch.Tensor(data[2]).to(device)

            model_open[i] = net(img, forces).cpu().data.numpy() > 0.5
            ground_truth_open[i] = ground_truth.cpu().data.numpy()

            print(f"{i+1}/{test_length} complete")

    # Calculate statistics
    correct_count = np.count_nonzero(model_open == ground_truth_open)
    accuracy = correct_count / len(ground_truth_open)

    actual_positives = ground_truth_open == True
    actual_negatives = ground_truth_open == False
    predicted_positives = model_open == True
    predicted_negatives = model_open == False

    true_positives = predicted_positives & actual_positives
    # true_negatives = predicted_negatives & actual_negatives
    false_positives = predicted_positives & actual_negatives
    false_negatives = predicted_negatives & actual_positives

    # ap = np.count_nonzero(actual_positives)
    # an = np.count_nonzero(actual_negatives)
    # pp = np.count_nonzero(predicted_positives)
    # pn = np.count_nonzero(predicted_negatives)
    tp = np.count_nonzero(true_positives)
    # tn = np.count_nonzero(true_negatives)
    fp = np.count_nonzero(false_positives)
    fn = np.count_nonzero(false_negatives)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2*(precision*recall)/(precision+recall)

    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    stats = {
        'model_name': model_name, 
        'data_session_id': session_id,
        'test_timestamp': timestamp ,
        'test_fraction': test_fraction,
        'using_segmentation': os.environ.get('use_segmentation') is not None,
        'accuracy': accuracy,
        'f_score': f_score
    }
    print(json.dumps(stats, indent=4))

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Evaluate model on dataset')
    parser.add_argument('--data', type=str, help='path of csv file to train on e.g. data/2021-12-09-04:56:05/raw.csv')
    parser.add_argument('--model-name', type=str, required=True, help='model file stored in top level models/ e.g. 2021-12-14-23.pt')

    current_dirname = os.path.dirname(__file__)
    params_path = os.path.join(current_dirname, 'params.yaml')
    with open(params_path, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

        args = parser.parse_args()
        params['data_file'] = args.data

        params = namedtuple("Params", params.keys())(*params.values())
        main(params, args.model_name)
    # example:
    # python3 evaluate_model.py --model-name 2021-12-14-23.pt --data data/2021-12-14-23/2021-12-14-23_calib_short.csv
    # model must be in models directory specified in params.yaml
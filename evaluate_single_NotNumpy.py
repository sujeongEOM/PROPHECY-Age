'''
This is for UKBiobank, Shaoxing, PTB-XL Dataset
'''

# Imports
from models.resnet_original import ResNet1d
from data.CustomDataset_NotNumpy import CustomDataset
import tqdm
import torch
import os
import json
import numpy as np
import argparse
from warnings import warn
import pandas as pd


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model_path', default="model_hx/", 
                        help='folder containing model')
    parser.add_argument('--pred_folder', type=str, default="model_hx/predicted-ages/", 
                        help='folder to save predicted ages')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='number of exams per batch.')
#    parser.add_argument('--output', type=str, default='predicted_age.csv',
#                        help='output file.')
#    parser.add_argument('--traces_dset', default='tracings',
#                         help='traces dataset in the hdf5 file.')
#    parser.add_argument('--ids_dset',
#                         help='ids dataset in the hdf5 file.')
    parser.add_argument('--model_name', default="220620-Single-2", 
                        help='name of the model')
    parser.add_argument('path_to_test_csv',  
                        help='path to csv containing age')
    parser.add_argument('path_to_test_traces', 
                        help='path to numpy containing ECG traces')
    parser.add_argument('dataset_name', type=str, 
                        help='dataset being tested')
    parser.add_argument('fname_col', 
                        help='column with the fname in csv file.')             

    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    if not os.path.exists(args.pred_folder):
        os.makedirs(args.pred_folder)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Get checkpoint
    ckpt = torch.load(os.path.join(args.model_path, f'{args.model_name}_model.pth'), map_location=lambda storage, loc: storage)
    # Get config
    config = os.path.join(args.model_path, f'{args.model_name}_args.json')
    with open(config, 'r') as f:
        config_dict = json.load(f)
    # Get model
    N_LEADS = 8
    model = ResNet1d(input_dim=(N_LEADS, config_dict['seq_length']),
                     blocks_dim=list(zip(config_dict['net_filter_size'], config_dict['net_seq_lengh'])),
                     n_classes=1,
                     kernel_size=config_dict['kernel_size'],
                     dropout_rate=config_dict['dropout_rate'])
    # load model checkpoint
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    # Get traces
    if args.dataset_name == "Shaoxing":
        test_df = pd.read_excel(args.path_to_test_csv)
    else:
        test_df = pd.read_csv(args.path_to_test_csv)
    test_fname = test_df[args.fname_col].to_numpy() + ".csv"

    n_total = len(test_df)

    dataset = CustomDataset(args.path_to_test_traces, test_fname)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    '''
    if args.ids_dset:
        ids = ff[args.ids_dset]
    else:
        ids = range(n_total)
    '''
    # Get dimension
    #predicted_age = np.zeros((n_total,))
    # Evaluate on test data
    model.eval()
    n_batches = int(np.ceil(n_total/args.batch_size))
    # Compute gradients
    predicted_age = np.zeros((n_total,))
    end = 0
    i=0
    for i, traces in enumerate(tqdm.tqdm(dataloader)):
        start = end
        end = min((i + 1) * args.batch_size, n_total)
        traces = traces.transpose(1, 2)
        traces = traces.to(device)
        with torch.no_grad():
            y_pred = model(traces)
        predicted_age[start:end] = y_pred.detach().cpu().numpy().flatten()
    # Save predictions
    test_df["predicted_age"] = predicted_age
    test_df.to_csv(os.path.join(args.pred_folder, f'{args.model_name}_{args.dataset_name}_predicted-age.csv'), index=False)
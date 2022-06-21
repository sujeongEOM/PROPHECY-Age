# Imports
from models.resnet_original import ResNet1d
import tqdm
import torch
import os
import json
import numpy as np
import argparse
from warnings import warn
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model_path', default="model_hx/", 
                        help='folder containing model')
    parser.add_argument('--pred_folder', type=str, default="model_hx/predicted-ages/", 
                        help='folder to save predicted ages')
    parser.add_argument('--age_col', default='RestingECG.PatientDemographics.PatientAge',
                        help='column with the age in csv file.')
    parser.add_argument('--fname_col', default='fname',
                        help='column with the fname in csv file.')             
    parser.add_argument('--batch_size', type=int, default=128,
                        help='number of exams per batch.')
#    parser.add_argument('--output', type=str, default='predicted_age.csv',
#                        help='output file.')
#    parser.add_argument('--traces_dset', default='tracings',
#                         help='traces dataset in the hdf5 file.')
#    parser.add_argument('--ids_dset',
#                         help='ids dataset in the hdf5 file.')
    parser.add_argument('model_name', 
                        help='name of the model')
    parser.add_argument('path_to_test_csv',  
                        help='path to csv containing age')
    parser.add_argument('path_to_test_traces', 
                        help='path to numpy containing ECG traces')
    parser.add_argument('dataset_name', type=str, 
                        help='dataset being tested')
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
    test_df = pd.read_csv(args.path_to_test_csv)
    test_ages = test_df[args.age_col]
    test_fname = test_df[args.fname_col]

    test_traces = np.load(args.path_to_test_traces, "r+")
    n_total = len(test_traces)
    print(f'traces shape: {test_traces.shape}')
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
    n_total, n_samples, n_leads = test_traces.shape
    n_batches = int(np.ceil(n_total/args.batch_size))
    # Compute gradients
    predicted_age = np.zeros((n_total,))
    end = 0
    for i in tqdm.tqdm(range(n_batches)):
        start = end
        end = min((i + 1) * args.batch_size, n_total)
        with torch.no_grad():
            x = torch.tensor(test_traces[start:end, :, :]).transpose(-1, -2) #(batch_size, 5120, 8)
            x = x.to(device, dtype=torch.float32)
            y_pred = model(x)
        predicted_age[start:end] = y_pred.detach().cpu().numpy().flatten()
    # Save predictions
    df = pd.DataFrame({'fname': test_fname, 'predicted_age': predicted_age})
    df.to_csv(os.path.join(args.pred_folder, f'{args.model_name}_{args.dataset_name}_predicted-age.csv'), index=False)
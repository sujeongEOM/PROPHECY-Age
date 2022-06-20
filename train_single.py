import json
import torch
import os
from tqdm import tqdm
from resnet import ResNet1d
from dataset import CustomDataset
import torch.optim as optim
import numpy as np


# Ribeiro
def compute_loss(ages, pred_ages):
    diff = ages.flatten() - pred_ages.flatten()
    loss = torch.sum(diff * diff)
    return loss

'''
# Deep Ensembles
def compute_loss(ages, pred_ages, pred_sigma):
  gauss_loss = torch.sum(0.5*torch.log(pred_sigma.flatten()) + 0.5*torch.div(torch.square(ages - pred_ages.flatten()), pred_sigma.flatten())) + 1e-6
  return gauss_loss
'''

'''
# Ribeiro
def compute_weights(ages, max_weight=np.inf):
    _, inverse, counts = np.unique(ages, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    normalized_weights = weights / sum(weights)
    w = len(ages) * normalized_weights
    # Truncate weights to a maximum
    if max_weight < np.inf:
        w = np.minimum(w, max_weight)
        w = len(ages) * w / sum(w)
    return w
'''


def train(ep, dataload):
    model.train()
    total_loss = 0
    n_entries = 0
    train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=len(dataload),
                     desc=train_desc.format(ep, 0, 0), position=0)
    for traces, ages in dataload:
        traces = traces.transpose(1, 2)
        traces, ages = traces.to(device), ages.to(device) #traces [128, 8, 5120] ages[128]
        model.zero_grad()
        # Send to device
        # Forward pass
        pred_ages = model(traces) #pred_ages [128, 1]
        #print(pred_ages, pred_ages.shape, type(pred_ages))
        #print(pred_sigma, pred_sigma.shape, type(pred_sigma))
        loss = compute_loss(ages, pred_ages)
        #print(loss, loss.shape)
        #print(loss.shape)  
        # Backward pass
        loss.backward() 
        # Optimize
        optimizer.step()
        # Update
        bs = len(traces)
        total_loss += loss.detach().cpu().numpy()
        n_entries += bs
        # Update train bar
        train_bar.desc = train_desc.format(ep, total_loss / n_entries)
        train_bar.update(1)
    train_bar.close()
    return total_loss / n_entries


def eval(ep, dataload):
    model.eval()
    total_loss = 0
    n_entries = 0
    eval_desc = "Epoch {:2d}: valid - Loss: {:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(dataload),
                    desc=eval_desc.format(ep, 0, 0), position=0)
    for traces, ages in dataload:
        traces = traces.transpose(1, 2)
        traces, ages = traces.to(device), ages.to(device)
        with torch.no_grad():
            # Forward pass
            pred_ages = model(traces)
            loss = compute_loss(ages, pred_ages)
            # Update outputs
            bs = len(traces)
            # Update ids
            total_loss += loss.detach().cpu().numpy()
            n_entries += bs
            # Print result
            eval_bar.desc = eval_desc.format(ep, total_loss / n_entries)
            eval_bar.update(1)
    eval_bar.close()
    return total_loss / n_entries


if __name__ == "__main__":
    import pandas as pd
    import argparse
    from torch.utils.data import DataLoader
    from warnings import warn
    import wandb

    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Train model to predict rage from the raw ecg tracing.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='maximum number of epochs (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for number generator (default: 42)')
    parser.add_argument('--sample_freq', type=int, default=500,
                        help='sample frequency (in Hz) in which all traces will be resampled at (default: 500)')
    parser.add_argument('--seq_length', type=int, default=5120,
                        help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                                    'to fit into the given size. (default: 5120)')
    #parser.add_argument('--scale_multiplier', type=int, default=10,
    #                    help='multiplicative factor used to rescale inputs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 128).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument("--patience", type=int, default=7,
                        help='maximum number of epochs without reducing the learning rate (default: 7)')
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help='minimum learning rate (default: 1e-7)')
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help='reducing factor for the lr in a plateu (default: 0.1)')
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320],
                        help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[5120, 1280, 320, 80, 20],
                        help='number of samples per resnet layer (default: [5120, 1280, 320, 80, 20]).')
    parser.add_argument('--dropout_rate', type=float, default=0.8,
                        help='dropout rate (default: 0.8).')
    parser.add_argument('--kernel_size', type=int, default=17,
                        help='kernel size in convolutional layers (default: 17).')
    parser.add_argument('--folder', default='model_hx/',
                        help='output folder (default: ./out)')
    #parser.add_argument('--traces_dset', default='tracings',
    #                    help='traces dataset in the hdf5 file.')
    #parser.add_argument('--ids_dset', default='',
    #                    help='by default consider the ids are just the order')
    parser.add_argument('--age_col', default='RestingECG.PatientDemographics.PatientAge',
                        help='column with the age in csv file.')
#   parser.add_argument('--ids_col', default=None,
#                        help='column with the ids in csv file.')
    parser.add_argument('--cuda', default=True,
                        help='use cuda for computations. (default: True)')
#    parser.add_argument('--n_valid', type=int, default=100,
#                        help='the first `n_valid` exams in the hdf will be for validation.'
#                             'The rest is for training')
    parser.add_argument('model_name',
                        help='name of the model')
    parser.add_argument('path_to_train_traces',
                        help='path to train file containing ECG traces') #train_np.npy
    parser.add_argument('path_to_train_csv',
                        help='path to train csv file containing attributes.') #220616_Train_Age-fname.csv
    parser.add_argument('path_to_valid_traces',
                        help='path to valid file containing ECG traces') #train_np.npy
    parser.add_argument('path_to_valid_csv',
                        help='path to valid csv file containing attributes.') #220616_Train_Age-fname.csv
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    torch.manual_seed(args.seed)
    print(args)

    wandb.login
    # wandb config
    wandb.init(
        project="PROPHECY-Age", entitiy="sjeom", 
        config={
            "model_name": args.model_name, 
            "epochs": args.epochs, 
            "batch_size": args.batch_size, 
            "lr": args.lr, 
            "dropout_rate": args.dropout_rate, 
            "kernel_size": args.kernel_size
        })
    
    config = wandb.config

    # Set device
    device = torch.device('cuda:0' if args.cuda else 'cpu')
    folder = args.folder

    # Generate output folder if needed
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)

    # Save config file
    with open(os.path.join(args.folder, f'{config.model_name}_args.json'), 'w') as f: #name edit
        json.dump(vars(args), f, indent='\t')

    tqdm.write("Building data loaders...")
    # train
    # Get csv data
    train_df = pd.read_csv(args.path_to_train_csv)
    train_ages = train_df[args.age_col]
    # Get h5 data
    train_traces = np.load(args.path_to_train_traces, "r+")
    # weights
    #train_weights = compute_weights(train_ages)

    # valid
    # Get csv data
    valid_df = pd.read_csv(args.path_to_valid_csv)
    valid_ages = valid_df[args.age_col]
    # Get h5 data
    valid_traces = np.load(args.path_to_valid_traces, "r+")
    # weights
    #valid_weights = compute_weights(valid_ages)

    # Dataset and Dataloader
    train_dataset = CustomDataset(train_traces, train_ages)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    valid_dataset = CustomDataset(valid_traces, valid_ages)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    tqdm.write("Done!")

    tqdm.write("Define model...")
    N_LEADS = 8  # the 8 leads
    N_CLASSES = 1  # just the age
    model = ResNet1d(input_dim=(N_LEADS, args.seq_length),
                     blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh)),
                     n_classes=N_CLASSES,
                     kernel_size=config.kernel_size,
                     dropout_rate=config.dropout_rate)
    model.to(device=device)
    print(model)
    tqdm.write("Done!")

    tqdm.write("Define optimizer...")
    optimizer = optim.Adam(model.parameters(), config.lr)
    tqdm.write("Done!")

    tqdm.write("Define scheduler...")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience,
                                                     min_lr=args.lr_factor * args.min_lr,
                                                     factor=args.lr_factor)
    tqdm.write("Done!")
    
    tqdm.write("Training...")
    start_epoch = 0
    best_loss = np.Inf
    history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr',
                                    'weighted_rmse', 'weighted_mae', 'rmse', 'mse'])
    for ep in range(start_epoch, config.epochs):
        train_loss = train(ep, train_loader)
        valid_loss = eval(ep, valid_loader)
        # Save best model
        if valid_loss < best_loss:
            # Save model
            torch.save({'epoch': ep,
                        'model': model.state_dict(),
                        'valid_loss': valid_loss,
                        'optimizer': optimizer.state_dict()},
                       os.path.join(folder, 'model.pth'))
            # Update best validation loss
            best_loss = valid_loss
        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Interrupt for minimum learning rate
        if learning_rate < args.min_lr:
            break

        # Print message
        tqdm.write('Epoch {:2d}: \tTrain Loss {:.6f} ' \
                  '\tValid Loss {:.6f} \tLearning Rate {:.7f}\t'
                 .format(ep, train_loss, valid_loss, learning_rate))
        
        # Wandb log
        metrics = {
            "train/epoch": ep, 
            "train/train_loss": train_loss, 
            "val/val_loss": valid_loss,
            "learning_rate": learning_rate
            }
            
        wandb.log(metrics)

        # Save history
        history = history.append({"epoch": ep, "train_loss": train_loss,
                                  "valid_loss": valid_loss, "lr": learning_rate}, ignore_index=True)
        history.to_csv(os.path.join(folder, f'{config.model_name}_history.csv'), index=False)
        # Update learning rate
        scheduler.step(valid_loss)
    wandb.finish()
    tqdm.write("Done!")



import os
import pandas as pd
from glob import glob
import numpy as np
import pickle

from tensorflow.keras.utils import Sequence
import tensorflow as tf


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, csv_files, labels, batch_size, shape=(5120, 8), shuffle=False):
        'Initialization'
        self.csv_files = csv_files
        self.labels = labels #df
        self.batch_size = batch_size
        self.shape = shape 
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.csv_files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        csv_batch = [self.csv_files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(csv_batch)
        
        # to track prediction progress
        print(f'{index} prediction start')

        return X, y

    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.csv_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    

    def __data_generation(self, csv_batch):
        '데이터 전처리하는 부분'

        X = np.empty((self.batch_size, *self.shape))
        y = np.empty((self.batch_size), dtype=int)
        
        # Lead zero padded to 5120
        # ECG Lead 12 to 8        
        for i, file in enumerate(csv_batch):            
            x = pd.read_csv(file)
            try:
                x.pop('III')
                x.pop('aVR')
                x.pop('aVL')
                x.pop('aVF')
                x_np = x.to_numpy()
                x_pad = np.pad(x_np, ((120,0),(0,0)),'constant',constant_values=0)
                x_pad = x_pad / 1000
                assert x_pad.shape == (5120, 8)
                X[i, ] = x_pad
            except:
                print(file)
            
            fname = file.split('/')[-1].split('.')[0]
            y[i] = self.labels[fname]

        return X, y
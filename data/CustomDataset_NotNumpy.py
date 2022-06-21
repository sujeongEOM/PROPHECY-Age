from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import numpy as np
#https://honeyjamtech.tistory.com/68
# Dataset 상속
class CustomDataset(Dataset): 
  def __init__(self, wavef_dir, fnames):
    self.wavef_dir = wavef_dir
    self.fnames = fnames #fname.csv
    
  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.fnames)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx):
    x = pd.read_csv(os.path.join(self.wavef_dir, self.fnames[idx]))
    x.pop("III")
    x.pop("aVR")
    x.pop("aVL")
    x.pop("aVF")
    x = x.to_numpy()
    x = np.pad(x, ((60,60),(0,0)),'constant',constant_values=0)
    x = x / 1000
    assert x.shape == (5120, 8)
    x = torch.Tensor(x)
    return x
from torch.utils.data import Dataset
import torch

#https://honeyjamtech.tistory.com/68
# Dataset 상속
class CustomDataset(Dataset): 
  def __init__(self, traces, ages):
    self.traces = traces
    self.ages = ages
    
  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.traces)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    trace = self.traces[idx]
    age = self.ages[idx]    
    assert trace.shape == (5120, 8)
    trace = torch.Tensor(trace)
    return trace, age
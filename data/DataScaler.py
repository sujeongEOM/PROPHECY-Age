import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scaler_transform(data, scaler): #need to fit on train first
    d1, d2, d3 = data.shape
    data = data.reshape(-1, d3)
    data = scaler.transform(data).reshape(d1, d2, d3)
    return data

train_np = np.load("/home/ubuntu/dr-you-ecg-20220420_mount/220616_Age_Dataset_sjeom/220616_train-np.npy")d

print(np.min(train_np), np.max(train_np))

scaler = MinMaxScaler()

# fit & transform on train data (ct1)
d1, d2, d3 = train_np.shape
train_np = train_np.reshape(-1, d3)
scaler.fit(train_np)
train_np = scaler.transform(train_np).reshape(d1, d2, d3)

print(np.min(train_scaled), np.max(train_scaled))
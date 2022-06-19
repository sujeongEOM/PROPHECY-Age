from tqdm import tqdm
import pandas as pd
import numpy as np

list_path = "/home/ubuntu/sjeom/ECG_Extract_Wave/Sev-MUSE-EKG/"
data_path = "/home/ubuntu/dr-you-ecg-20220420_mount/220425_SevMUSE_EKG_waveform/"

train_df = pd.read_csv(list_path + "220616_Train_Age-fname.csv")


train_list = []
for i in tqdm(range(len(train_df))):
    wf_path = data_path + train_df["fname"][i] + ".csv"
    x = pd.read_csv(wf_path)

    x.pop('III')
    x.pop('aVR')
    x.pop('aVL')
    x.pop('aVF')
    x_np = x.to_numpy()
    x_pad = np.pad(x_np, ((60,60),(0,0)),'constant',constant_values=0)
    x_pad = x_pad / 1000
    assert x_pad.shape == (5120, 8)
    train_list.append(x_pad)

train_list = np.array(train_list)
np.save("220616_train-np", train_list)
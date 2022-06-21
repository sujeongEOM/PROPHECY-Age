from tqdm import tqdm
import pandas as pd
import numpy as np

list_path = "/home/ubuntu/sjeom/ECG_Extract_Wave/Sev-MUSE-EKG/"
data_path = "/home/ubuntu/dr-you-ecg-20220420_mount/220425_SevMUSE_EKG_waveform/"

test_df = pd.read_csv(list_path + "220602_IntTest_Age-fname.csv")
print(len(test_df))
df_split = np.split(test_df, 3)

for n, df in enumerate(df_split):
    test_list = []
    df = df.reset_index()
    df.to_csv(f"220616_test-df_{n}.csv")
    for i in tqdm(range(len(df))):
        wf_path = data_path + df["fname"][i] + ".csv"
        x = pd.read_csv(wf_path)

        x.pop('III')
        x.pop('aVR')
        x.pop('aVL')
        x.pop('aVF')
        x_np = x.to_numpy()
        x_pad = np.pad(x_np, ((60,60),(0,0)),'constant',constant_values=0)
        x_pad = x_pad / 1000
        assert x_pad.shape == (5120, 8)
        test_list.append(x_pad)

    test_list = np.array(test_list)
    np.save(f"/home/ubuntu/dr-you-ecg-20220420_mount/220616_Age_Dataset_sjeom/220616_test-np_{n}", test_list)
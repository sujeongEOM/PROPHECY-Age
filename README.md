# PROPHECY-Age
project start on 2022.03	   
Goal : Predicting age using EKG waveform data
     
     
# Process of EKG projects
1. Extracting waveform, label table from XML or tsv; parsing and decoding
2. Data preprocessing (check waveform shape, label null etc.)
3. Model train, test        

  
  
## 1. Extraction difference with Dataset
> 1. Severance MUSE EKG tsv (Train / Internal Validation)
-- Whole data in huge single tsv file (220GB)
- Each row equals one EKG file
- There are pt overlap (as single pt takes many EKGs)
- Label Table : Whole tsv file saved in csv only with WaveformData column deleted
- Waveform : WaveformData column needs base64 decoding; decoded and saved into a single csv file  



> 2. UK Biobank (External Validation)
- XML data
- Need to parse each XML file
- Single XML file equals one EKG file
- Parsing function on branch _**UK Biobank**_



## 2. Data Preprocessing
> 1. Waveform Data
- 500Hz / 10 sec / 12 leads
- shape = (5000, 12)
- Use only 8 leads (I, II, V1-V6) as other 4 leads are computed using these 8 leads
- 0 padding on top 120 rows
- final shape as input = (5120, 12)


## 3. Model Train, Test
- Data Generator
- 1D CNN model (Ribeiro, Attia etc.)
- Test values : MSE, MAE, r2 score

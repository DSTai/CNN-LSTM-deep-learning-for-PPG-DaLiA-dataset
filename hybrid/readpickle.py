import pandas as pd
import numpy as np
import sys
import argparse

# Constants
RESAMPLE_CHEST = 175
RESAMPLE_WRIST = 8
LABELS = [1, 2, 3, 4, 5, 6, 7, 8]

def read_data(filepath):
    print("Reading data from:", filepath)
    df = pd.read_pickle(filepath)

    label = []
    CHEST_ACC_X = []
    CHEST_ACC_Y = []
    CHEST_ACC_Z = []
    WRIST_ACC_X = []
    WRIST_ACC_Y = []
    WRIST_ACC_Z = []

    list_label = df['activity'].tolist()
    list_chest_ACC = df['signal']['chest']['ACC'].tolist()
    list_wrist_ACC = df['signal']['wrist']['ACC'].tolist()

    print('Length of activity:', len(list_label))
    for i in range(0, int(len(list_chest_ACC) / RESAMPLE_CHEST)):
        CHEST_ACC_X.append(list_chest_ACC[i * RESAMPLE_CHEST][0])
        CHEST_ACC_Y.append(list_chest_ACC[i * RESAMPLE_CHEST][1])
        CHEST_ACC_Z.append(list_chest_ACC[i * RESAMPLE_CHEST][2])

    for l in range(0, int(len(list_wrist_ACC) / RESAMPLE_WRIST)):
        WRIST_ACC_X.append(list_wrist_ACC[l * RESAMPLE_WRIST][0])
        WRIST_ACC_Y.append(list_wrist_ACC[l * RESAMPLE_WRIST][1])
        WRIST_ACC_Z.append(list_wrist_ACC[l * RESAMPLE_WRIST][2])

    count_label = [0] * len(LABELS)
    for j in range(0, int(len(list_label))):
        label_val = int(list_label[j][0])  # Convert label to integer
        if label_val in LABELS:
            count_label[label_val - 1] += 1
            label.append(label_val)
      #  else:
         #   print("Invalid label value:", label_val)

    print('Count of labels:', count_label)
    print('Length of CHEST-ACC-X:', len(CHEST_ACC_X))
    print('Length of CHEST-ACC-Y:', len(CHEST_ACC_Y))
    print('Length of CHEST-ACC-Z:', len(CHEST_ACC_Z))
    print('Length of WRIST-ACC-X:', len(WRIST_ACC_X))
    print('Length of WRIST-ACC-Y:', len(WRIST_ACC_Y))
    print('Length of WRIST-ACC-Z:', len(WRIST_ACC_Z))
    print('Length of activity:', len(label))

    df_fn = pd.DataFrame(list(zip(label,
                                  CHEST_ACC_X,
                                  CHEST_ACC_Y,
                                  CHEST_ACC_Z,
                                  WRIST_ACC_X,
                                  WRIST_ACC_Y,
                                  WRIST_ACC_Z)),
                         columns=['label',
                                  'CHEST-ACC-X',
                                  'CHEST-ACC-Y',
                                  'CHEST-ACC-Z',
                                  'WRIST-ACC-X',
                                  'WRIST-ACC-Y',
                                  'WRIST-ACC-Z'])

    print('Finished loading data...')
    return df_fn


def summarize_data(df):
    print('Summary of the DataFrame:')
    print(df.describe())
    print('DataFrame Head:')
    print(df.head())



parser = argparse.ArgumentParser(description="Process and summarize a pickle file")
parser.add_argument("--pickle_file",default="D:\Documents\Downloads\Human_Activity_Recognition\DALIA\\train\S7.pkl", type=str, required=False, help="Path to the pickle file")
args = parser.parse_args()

df = read_data(args.pickle_file)
summarize_data(df)

#python readpickle.py --pickle_file D:/Documents/Downloads/Human_Activity_Recognition/DALIA/train/S7.pkl
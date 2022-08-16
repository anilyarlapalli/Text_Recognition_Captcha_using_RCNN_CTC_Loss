
from cProfile import label
import os
import glob
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import dataset
import pandas as pd


def encode_data():
    df = pd.read_excel(r"input/labels/captcha_labels.xlsx", engine='openpyxl')

    image_files = df["image_filepaths"]
    image_files = np.array(image_files)

    targets_orig =  df["image_labels"]
    targets = [[c for c in str(x)] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]

    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
    targets_enc = [lbl_enc.transform(x) for x in targets]

    torch_enc = [torch.Tensor(x) for x in targets_enc]
    labels_padded = pad_sequence(torch_enc, padding_value = 0.)
    labels_padded = torch.transpose(labels_padded, 1, 0)

    labels_padded = np.array(labels_padded, dtype = int)
    print(f"No of classes: {len(lbl_enc.classes_)}\n, Length of each target sequence: {len(labels_padded[0])}")
    # print("target sequences sample: ", labels_padded[:5])
    # target sequence_length = 16 i.e max len, len(lbl_enc.classes_) = 108
    df = pd.DataFrame(lbl_enc.classes_)
    df.to_csv('preds/label_classes.csv',encoding='utf-8-sig', index = False) 


    return(image_files, labels_padded, targets_orig, lbl_enc)

if __name__ == "__main__":
    encode_data()

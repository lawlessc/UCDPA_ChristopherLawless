
import pandas as pd


IDS_AND_TARGETS =  pd.read_csv("data/training_labels.csv")

def import_targets():
    targets_df =  pd.read_csv("data/training_labels.csv")
    IDS_AND_TARGETS = targets_df
    #print(type(targets_df))
    #print(targets_df.shape)
    #print(targets_df.head())
    return targets_df




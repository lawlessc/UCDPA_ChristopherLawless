
import pandas as pd


#IDS_AND_TARGETS =  pd.read_csv("data/training_labels.csv")

def import_training_targets():
    """running this imports a dataframe of all the  training sample ID's and their target values"""
    #print("import_training_targets")
    targets_df =  pd.read_csv("data/training_labels.csv",dtype={"id": "string", "target": float} ,low_memory=False,delimiter=",")
    #IDS_AND_TARGETS = targets_df
    #print(type(targets_df))
    #print(targets_df.shape)
    #print(targets_df.head())
    #print("END import_training_targets")
    return targets_df

def import_negative_training_targets():
    """running this imports a dataframe of all the  training sample ID's and their target values"""
    #print("import_training_targets")
    targets_df =  pd.read_csv("data/training_labels_no_blackholes.csv",low_memory=False,delimiter=',')
    #IDS_AND_TARGETS = targets_df
    #print(type(targets_df))
    #print(targets_df.shape)
    #print(targets_df.head())
    #print("END import_training_targets")
    return targets_df

def import_positive_training_targets():
    """running this imports a dataframe of all the  training sample ID's and their target values"""
    #print("import_training_targets")
    targets_df =  pd.read_csv("data/training_labels_with_blackholes.csv",low_memory=False,delimiter=',')
    #IDS_AND_TARGETS = targets_df
    #print(type(targets_df))
    #print(targets_df.shape)
    #print(targets_df.head())
    #print("END import_training_targets")
    return targets_df



def import_testing_targets():
    """running this imports a dataframe of all the training ID's and their target values
    (these targets aren't actually relevant but are just for showing submission formatting for the Kaggle competition)"""
    targets_df =  pd.read_csv("data/sample_submission.csv")
    return targets_df




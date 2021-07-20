import pandas as pd
import numpy as np
import get_ligo_path as gp
import import_targets as tg

#just a test on a single file, shows 3 rows, 4096 colums
#These rows correspond to the 3 detectors.
def import_test():
    testnpy = np.load("data/00000e74ad.npy")
    print(type(testnpy))
    print(testnpy.shape)
    return testnpy


def import_to_dataframe_test():
    testnpy = np.load("00000e74ad")
    #df = pd.DataFrame(testnpy,columns =["detector_1","detector_2","detector_3"])
    df = pd.DataFrame(testnpy).T #Transposed to avoid awefulness.
    return  df


def import_single_sample(sample_name):
    #samples names are hex numbers in the targets file
    #The first 3 characters represent
    #print(gp.get_path(sample_name))
    #print(gp.get_path_training(sample_name))

    testnpy = np.load(gp.get_path_training(sample_name))
    #df = pd.DataFrame(testnpy).T
    df = pd.DataFrame(testnpy)

def import_flat_single_sample(sample_name):
    #samples names are hex numbers in the targets file
    #The first 3 also represent the path characters represent
    testnpy = np.load(gp.get_path_training(sample_name))

    testnpy_flat= testnpy.flatten()

    #flatten to a single row
    df = pd.DataFrame(testnpy_flat).T

    print(df)
    print(type(df))
    print(df.head())
    return df

#this takes a list of sample names and imports them
def import_list_of_flat_samples(sample_name_list):
    #We create an empty dataframe
    df = pd.DataFrame()

    for sample in sample_name_list:
        df = df.append(import_flat_single_sample(sample))

    #print(df)
    #print(type(df))
    #print(df.head())
    return df

#by selecting a starting point and an end point the user can select a range of samples
def import_many_flat_samples(starting_number, ending_number):
    targets_df = tg.import_targets()
    print(targets_df.head())

    sample_name_list = []

    # get the list of sample names for the range
    for entry in range(starting_number,ending_number):
        sample_name_list.append(targets_df["id"].values[entry])

    print(sample_name_list)
    return import_list_of_flat_samples(sample_name_list)






def import_many_flat_samples_add_targets(starting_number, ending_number):
    targets_df = tg.import_targets()
    print(targets_df.head())

    sample_name_list = []
    target_list = []

    #get the list of sample names for the range
    for entry in range(starting_number,ending_number):
        sample_name_list.append(targets_df["id"].values[entry])
        target_list.append(targets_df["target"].values[entry])

    print(sample_name_list)
    samples_df = import_list_of_flat_samples(sample_name_list)
    samples_df["target"] = target_list
    print(samples_df)
    return samples_df



def import_single_samples_of_number(number):
    targets_df = tg.import_targets()
    return targets_df["id"].values[number]



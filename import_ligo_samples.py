import pandas as pd
import numpy as np
import get_ligo_path as gp
import import_targets as tg
from sklearn import preprocessing

#just a test on a single file, shows 3 rows, 4096 columns
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
    df = pd.DataFrame(testnpy)

def import_flat_training_sample(sample_name):
    #samples names are hex numbers in the targets file
    #The first 3 also represent the path characters represent
    testnpy = np.load(gp.get_path_training(sample_name))
             #This flattens a sample numpy from 3 rows to one row
    flat = testnpy.flatten()
    return flat

def import_flat_testing_sample(sample_name):

    testnpy = np.load(gp.get_path_testing(sample_name))
    flat = testnpy.flatten()
    return flat

#this takes a list of sample names and imports them
def import_list_of_flat_training_samples(sample_name_list):
    #We create an empty dataframe
    numpy_arrays_list = []
    df = pd.DataFrame()

    for sample in sample_name_list:
        #print("...")
        numpy_arrays_list.append(import_flat_training_sample(sample))

    #this appends all the rows of numpy arrays list to the dataframe
    df = df.append(pd.DataFrame(numpy_arrays_list))
    #df.append(pd.DataFrame(numpy_arrays_list))
    return df

def import_list_of_flat_testing_samples(sample_name_list):
    #We create an empty dataframe
    numpy_arrays_list = []
    df = pd.DataFrame()

    for sample in sample_name_list:
        numpy_arrays_list.append(import_flat_testing_sample(sample))
    #this appends all the rows of numpy arrays list to the dataframe
    df = df.append(pd.DataFrame(numpy_arrays_list))
    #df.append(pd.DataFrame(numpy_arrays_list))
    return df


def import_number_of_flat_testing_samples(starting_number, ending_number):
    targets_df = tg.import_testing_targets()
    sample_name_list = []

    # get the list of sample names for the range
    for entry in range(starting_number,ending_number):
        sample_name_list.append(targets_df["id"].values[entry])
    #print(sample_name_list)
    #return normalise_data( import_list_of_flat_testing_samples(sample_name_list) )
    return import_list_of_flat_testing_samples(sample_name_list)


#by selecting a starting point and an end point the user can select a range of samples
def import_number_of_flat_training_samples(starting_number, ending_number):
    targets_df = tg.import_training_targets()
    sample_name_list = []

    # get the list of sample names for the range
    for entry in range(starting_number,ending_number):
        sample_name_list.append(targets_df["id"].values[entry])
    #print(sample_name_list)
    #return normalise_data(import_list_of_flat_training_samples(sample_name_list))
    return import_list_of_flat_training_samples(sample_name_list)






def import_many_flat_samples_add_targets(starting_number, ending_number):
    print("Importing many samples")
    targets_df = tg.import_training_targets()
    #print(targets_df.head())

    sample_name_list = []
    target_list = []

    print("appending targets")
    #get the list of sample names for the range
    for entry in range(starting_number,ending_number):
        sample_name_list.append(targets_df["id"].values[entry])
        target_list.append(targets_df["target"].values[entry])

        #print(str(targets_df["id"].values[entry])+" "+ str(targets_df["target"].values[entry]))


    print("import_list_of_flat_samples")
    samples_df = import_list_of_flat_training_samples(sample_name_list)
    samples_df["target"] = target_list
    #print(samples_df)
    return samples_df



def import_single_samples_of_number(number):
    targets_df = tg.import_training_targets()
    return targets_df["id"].values[number]






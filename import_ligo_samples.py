import math
import pandas as pd
import numpy as np
import get_sample_path as gp
import import_targets as tg
import visualisers as vs
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler



# just a test on a single file, shows 3 rows, 4096 columns
# These rows correspond to the 3 detectors.
def import_test():
    """This just imports one specific hardcoded file and displays the three signals,to confirm things are working"""
    sample = np.load("data/train/0/0/0/00000e74ad.npy")
    print("import_test():")
    print(sample[0, 0])
    print(type(sample))
    print(sample.shape)
    signal_list = [sample[0, :], sample[1, :], sample[2, :]]
    vs.signal_plotter(signals_list=signal_list)
    vs.fftPlot(sample[0, :])
    vs.spectralplot_multiple(data=signal_list)
    return sample


def import_single_sample_as_ndarray(sample_name):
    """This imports a single sample as a ndarray"""
    detector_readings = np.load(gp.get_path_training(sample_name))
    return detector_readings


def import_single_sample_as_dataframe(sample_name):
    """This imports a single sample as a DataFrame"""
    detector_readings = np.load(gp.get_path_training(sample_name))
    df = pd.DataFrame(detector_readings)
    return df


def import_flat_training_sample(sample_name):
    """This imports a single sample from the training set (3 sensor readings) and returns it to the user as flat
    numpy array of lenght 12288 """
    # print("import_flat_training_sample")
    # samples names are hex numbers in the targets file
    # The first 3 also represent the path characters represent
    detector_readings = np.load(gp.get_path_training(sample_name), allow_pickle=True, fix_imports=True).astype(
        np.float32)  # Added these to make sure the data wasn't being loaded mangled somehow
    # detector_readings[2, :] = np.negative(detector_readings[2, :])#i inverted this signal as it's on the otherside of the world to the other detectors.
    # print(detector_readings)
    return detector_readings.flatten()


def import_flat_testing_sample(sample_name):
    """This imports a single sample as a numpy array from the competitions testing set"""
    detector_readings = np.load(gp.get_path_testing(sample_name))
    return detector_readings.flatten()


# this takes a list of sample names and imports them
def import_list_of_flat_training_samples(sample_name_list):
    """This takes flat samples as an array of ID's from the training set"""
    numpy_arrays_list = []
    df = pd.DataFrame()
    for sample in sample_name_list:
        numpy_arrays_list.append(import_flat_training_sample(sample))
    # this appends all the rows of numpy arrays list to the dataframe

    # print(numpy_arrays_list)
    df = df.append(pd.DataFrame(numpy_arrays_list))
    # print(df)
    # df.append(pd.DataFrame(numpy_arrays_list))
    return df


def import_list_of_flat_testing_samples(sample_name_list):
    """This takes flat samples as an array of ID's from the testing set"""
    numpy_arrays_list = []
    df = pd.DataFrame()
    for sample in sample_name_list:
        numpy_arrays_list.append(import_flat_testing_sample(sample))
    # this appends all the rows of numpy arrays list to the dataframe
    df = df.append(pd.DataFrame(numpy_arrays_list))
    return df


def import_number_of_flat_testing_samples(starting_number, ending_number):
    """This takes takes integers,0 to N and imports them from the testing set"""
    targets_df = tg.import_testing_targets()
    sample_name_list = []
    # get the list of sample names for the range
    for entry in range(starting_number, ending_number + 1):
        sample_name_list.append(targets_df["id"].values[entry])
    return import_list_of_flat_testing_samples(sample_name_list)


def import_flat_samples_add_targets(starting_number, ending_number):
    """This takes takes integers,0 to N and imports them from the training set as panda, it also adds the targets and
    ID's to them """
    targets_df = tg.import_training_targets()
    sample_name_list = []
    target_list = []

    for entry in range(starting_number, ending_number + 1):
        sample_name_list.append(targets_df["id"].values[entry])
        target_list.append(float(targets_df["target"].values[entry]))

    samples_df = import_list_of_flat_training_samples(sample_name_list)
    samples_df["target"] = target_list
    samples_df["id"] = sample_name_list
    print(samples_df)
    return samples_df


def import_single_sample_number(number):
    """This takes a single number representing location of an Id in the training labels file """
    targets_df = tg.import_training_targets()
    return targets_df["id"].values[number]

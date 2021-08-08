import math

import pandas as pd
import numpy as np
import get_ligo_path as gp
import import_targets as tg
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MaxAbsScaler,MinMaxScaler,Normalizer, normalize, RobustScaler, scale
import visualisers as vs
from fractions import gcd
from scipy.fft import fft ,fftfreq , ifft , rfft, rfftfreq
from scipy.fft import irfft
#import scipy.stats.signaltonoise as sn
from sklearn.decomposition import PCA


import noisereduce as nr
#Added citation
# @software{tim_sainburg_2019_3243139,
#   author       = {Tim Sainburg},
#   title        = {timsainb/noisereduce: v1.0},
#   month        = jun,
#   year         = 2019,
#   publisher    = {Zenodo},
#   version      = {db94fe2},
#   doi          = {10.5281/zenodo.3243139},
#   url          = {https://doi.org/10.5281/zenodo.3243139}
# }



#just a test on a single file, shows 3 rows, 4096 columns
#These rows correspond to the 3 detectors.
def import_test():
    '''This just imports one specific hardcoded file to confirm things are working'''
    testnpy = np.load("data/train/0/0/0/00000e74ad.npy")
    print("import_test():")
    print(testnpy[0,0])
    print(type(testnpy))
    print(testnpy.shape)
    return testnpy

def import_single_sample(sample_name):
    '''This imports a single sample as a data frame'''
    detector_readings = np.load(gp.get_path_training(sample_name))
    df = pd.DataFrame(detector_readings)

def import_flat_training_sample(sample_name):
    '''This imports a single sample from the training set (3 sensor readings) and returns it to the user as flat numpy array of lenght 12288
    I have also attempted to do some preprocessing here'''
    #print("import_flat_training_sample")
    #samples names are hex numbers in the targets file
    #The first 3 also represent the path characters represent
    detector_readings = np.load(gp.get_path_training(sample_name),allow_pickle=True,fix_imports=True).astype(np.float32)
    #np.set_printoptions(threshold=np.inf)

    #print(detector_readings)

    #print(detector_readings)

    # predictor_scaler = StandardScaler().fit(detector_readings)
    # detector_readings = predictor_scaler.transform(detector_readings)

    # normalizer =   Normalizer().fit(np.array(detector_readings))
    # detector_readings = normalizer.transform(np.array(detector_readings))



    #detector_readings[2, :] = np.negative(detector_readings[2, :])
    # noise reduce--------------------------------------------------------
    # detector_readings = nr.reduce_noise(y=detector_readings, sr=5200, stationary=False)
    #detector_readings[0, :] = normalize(detector_readings[0, :])
    #detector_readings[1, :] = normalize(detector_readings[1, :])
    #detector_readings[2, :] = normalize(detector_readings[2, :])



    #normalizer =   Normalizer().fit(np.array(detector_readings))
    #detector_readings = normalizer.transform(np.array(detector_readings))


    #signals =[detector_readings[0, :],detector_readings[1, :],detector_readings[2, :]]
    #vs.signal_plotter(self=vs,signals_list=signals)

    # detector_mean = np.average([detector_readings[0, :], detector_readings[1, :], detector_readings[2, :]],
    #     axis =0 )#, dtype=np.float64)

    # signals = [detector_mean]
    # vs.signal_plotter(self=vs, signals_list=signals)

    #signals = [detector_readings[0, :], detector_readings[1, :], detector_readings[2, :]]
    #vs.signal_plotter(self=vs,signals_list=signals)


   # detector_mean = np.mean([detector_readings[0, :], detector_readings[1, :], detector_readings[2, :]],
                         #    axis =0 , dtype=np.float64)

    #detector_mean = np.average([detector_readings[0, :], detector_readings[1, :], detector_readings[2, :]],
    #                       axis=0)

    #vs.fftPlot(self=vs ,data=detector_readings[0,:])
    #vs.fftPlot(self=vs, data=detector_mean)



    #detector_mean = np.mean([detector_readings[0,:],detector_readings[1,:],detector_readings[2,:]],
                           # axis =0 , dtype=np.float64)







    #print(detector_mean)

   # print(len(detector_mean))

    #detector_mean =detector_mean.reshape(1,-1)

    #predictor_scaler = StandardScaler().fit(detector_mean)
    #detector_mean = predictor_scaler.transform(detector_mean)

    #predictor_scaler = StandardScaler().fit(detector_readings)
    #detector_readings = predictor_scaler.transform(detector_readings)


    #vs.spectralplot(self=vs,data=detector_mean)






    #predictor_scaler = StandardScaler().fit(testnpy)
    #testnpy = predictor_scaler.transform(testnpy)

   # robusts = RobustScaler()
   # testnpy = robusts.fit_transform(testnpy)


    #vs.fftPlot(self=vs,xf=xf1,yf=yf1)
    #print("x------------------")
    #signal_list = [detector_readings[0,:],detector_readings[1,:],detector_readings[2,:]]#,detector_mean]#,testnpy[2,:]]
    #vs.signal_plotter(self=vs ,signals_list=  signal_list)



    #signal_list = [testnpy[0,:]]
    #vs.signal_plotter(self=vs ,signals_list=  signal_list)

   # print(detector_readings.flatten())
   #print("return end:"+ str(return_value[12287]))

    #print("The readins")
    #print(detector_readings)


    #print("Flatten")
    #detector_readings= detector_readings.flatten()
    #print(str(detector_readings[0]))
    #print(str(detector_readings.shape[1]))

    #print(detector_readings)

    return detector_readings.flatten()

def import_flat_testing_sample(sample_name):
    '''This imports a single sample as a numpy array from the competitions testing set'''
    detector_readings = np.load(gp.get_path_testing(sample_name))
    return detector_readings.flatten()

#this takes a list of sample names and imports them
def import_list_of_flat_training_samples(sample_name_list):
    '''This takes flat samples as an array of ID's from the training set'''
    numpy_arrays_list = []
    df = pd.DataFrame()
    for sample in sample_name_list:
        numpy_arrays_list.append(import_flat_training_sample(sample))
    #this appends all the rows of numpy arrays list to the dataframe

    print(numpy_arrays_list)

    df = df.append(pd.DataFrame(numpy_arrays_list))
    print(df)
    #df.append(pd.DataFrame(numpy_arrays_list))
    return df

def import_list_of_flat_testing_samples(sample_name_list):
    '''This takes flat samples as an array of ID's from the testing set'''
    #create an empty dataframe
    numpy_arrays_list = []
    df = pd.DataFrame()

    for sample in sample_name_list:
        numpy_arrays_list.append(import_flat_testing_sample(sample))
    #this appends all the rows of numpy arrays list to the dataframe
    df = df.append(pd.DataFrame(numpy_arrays_list))
    return df


def import_number_of_flat_testing_samples(starting_number, ending_number):
    '''This takes takes integers,0 to N and imports them from the testing set'''
    targets_df = tg.import_testing_targets()
    sample_name_list = []
    # get the list of sample names for the range
    for entry in range(starting_number,ending_number+1):
        sample_name_list.append(targets_df["id"].values[entry])
    return import_list_of_flat_testing_samples(sample_name_list)

def import_flat_samples_add_targets(starting_number, ending_number):
    '''This takes takes integers,0 to N and imports them from the training set as panda, it also adds the targets and ID's to them'''
    targets_df = tg.import_training_targets()
    sample_name_list = []
    target_list = []


    for entry in range(starting_number,ending_number+1):
        sample_name_list.append(targets_df["id"].values[entry])
        #target_list.append(float(targets_df["target"].values[entry]))
        target_list.append(targets_df["target"].values[entry])

    samples_df = import_list_of_flat_training_samples(sample_name_list)
    #print(samples_df)
    samples_df["target"] = target_list
    samples_df["id"] = sample_name_list
    #print(samples_df)
    return samples_df



def import_single_sample_number(number):
    '''This takes a single number representing location of an Id in the training labels file '''
    targets_df = tg.import_training_targets()
    return targets_df["id"].values[number]



# def signal_process(sample_for_processing):
#     print("nothing yet")
#
#     y = rfft(sample_for_processing)
#     x = rfftfreq(4096, 1 / 2048)
#
#     points_per_freq = len(x) / (2048 / 2)
#     for frequency in range(0, 250):
#         target = int(points_per_freq * frequency)
#         y[target - 1: target + 2] = 0
#     y = irfft(y)
#
#     for frequency in range(450, 1800):
#         target = int(points_per_freq * frequency)
#         y[target - 1: target + 2] = 0
#     y = irfft(y)
#     return sample_for_processing
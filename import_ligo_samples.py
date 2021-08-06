import math

import pandas as pd
import numpy as np
import get_ligo_path as gp
import import_targets as tg
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MaxAbsScaler,MinMaxScaler,Normalizer, normalize, RobustScaler
import visualisers as vs
from fractions import gcd
from scipy.fft import fft ,fftfreq , ifft , rfft, rfftfreq
from scipy.fft import irfft
#import scipy.stats.signaltonoise as sn
from sklearn.decomposition import PCA



#just a test on a single file, shows 3 rows, 4096 columns
#These rows correspond to the 3 detectors.
def import_test():
    testnpy = np.load("data/train/0/0/0/00000e74ad.npy")
    #print("import_test():")

    #print(testnpy[0,0])
    #print(type(testnpy))

    #print(testnpy.shape)
    return testnpy

def import_single_sample(sample_name):
    testnpy = np.load(gp.get_path_training(sample_name))
    df = pd.DataFrame(testnpy)

def import_flat_training_sample(sample_name):
    #print("import_flat_training_sample")
    #samples names are hex numbers in the targets file
    #The first 3 also represent the path characters represent
    detector_readings = np.load(gp.get_path_training(sample_name))



    #doing this beacause i think scientific notation was messing with everything.
    #Manual scaling
    bignum  = 100000000000000000000
    bignum2 = 1000000000000000000000*0.4

    detector_readings[0,:] =   np.multiply(bignum, detector_readings[0,:])
    detector_readings[1, :] =  np.multiply(bignum, detector_readings[1, :])
    detector_readings[2, :] =  np.multiply(bignum2, detector_readings[2, :])

    #testnpy[2, :] = np.negative(testnpy[2, :])#maybe the location of the virgo array means flipping this will help
    detector_readings[2, :] = np.negative(detector_readings[2, :])
    #testnpy[1, :] = np.negative(testnpy[1, :])

    #predictor_scaler = StandardScaler().fit(detector_readings)
    #detector_readings = predictor_scaler.transform(detector_readings)

    vs.spectralplot(self=vs,data=detector_readings[1,:])




    #I learnt to use this here https://realpython.com/python-scipy-fft/
    # yf1=  rfft(detector_readings[0, :])
    # xf1 =rfftfreq(4096, 1 / 2048)
    # points_per_freq = len(xf1) / (2048 / 2)
    # for frequency in range(0,30):
    #     target_idx = int(points_per_freq * frequency)
    #     yf1[target_idx - 1: target_idx + 2] = 0
    # detector_readings[0, :]= irfft(yf1)
    #
    # for frequency in range(120,1100):
    #     target_idx = int(points_per_freq * frequency)
    #     yf1[target_idx - 1: target_idx + 2] = 0
    # detector_readings[0, :] = irfft(yf1)
    #
    # yf2=  rfft(detector_readings[1, :])
    # xf2 =rfftfreq(4096, 1 / 2048)
    # points_per_freq = len(xf2) / (2048 / 2)
    # for frequency in range(0,30):
    #     target_idx = int(points_per_freq * frequency)
    #     yf2[target_idx - 1: target_idx + 2] = 0
    # detector_readings[1, :]= irfft(yf2)
    #
    # for frequency in range(120,1100):
    #     target_idx = int(points_per_freq * frequency)
    #     yf2[target_idx - 1: target_idx + 2] = 0
    # detector_readings[1, :]= irfft(yf2)
    #
    # yf3=  rfft(detector_readings[2, :])
    # xf3 =rfftfreq(4096, 1 / 2048)
    # points_per_freq = len(xf3) / (2048 / 2)
    # for frequency in range(0,30):
    #     target_idx = int(points_per_freq * frequency)
    #     yf3[target_idx - 1: target_idx + 2] = 0
    # detector_readings[2, :] = irfft(yf3)
    #
    # for frequency in range(120,1100):
    #     target_idx = int(points_per_freq * frequency)
    #     yf3[target_idx - 1: target_idx + 2] = 0
    # detector_readings[2, :] = irfft(yf3)

    #predictor_scaler = StandardScaler().fit(testnpy)
    #testnpy = predictor_scaler.transform(testnpy)

   # robusts = RobustScaler()
   # testnpy = robusts.fit_transform(testnpy)


    #vs.fftPlot(self=vs,xf=xf1,yf=yf1)

    #signal_list = [detector_readings[0,:],detector_readings[1,:],detector_readings[2,:]]#,testnpy[1,:],testnpy[2,:]]
    #vs.signal_plotter(self=vs ,signals_list=  signal_list)



    #signal_list = [testnpy[0,:]]
    #vs.signal_plotter(self=vs ,signals_list=  signal_list)


   #print("return end:"+ str(return_value[12287]))
    return detector_readings.flatten()

def import_flat_testing_sample(sample_name):

    testnpy = np.load(gp.get_path_testing(sample_name))
    flat = testnpy.flatten()
    return flat

#this takes a list of sample names and imports them
def import_list_of_flat_training_samples(sample_name_list):
    numpy_arrays_list = []
    df = pd.DataFrame()


    for sample in sample_name_list:
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
    return df


def import_number_of_flat_testing_samples(starting_number, ending_number):
    targets_df = tg.import_testing_targets()
    sample_name_list = []
    # get the list of sample names for the range
    for entry in range(starting_number,ending_number):
        sample_name_list.append(targets_df["id"].values[entry])
    return import_list_of_flat_testing_samples(sample_name_list)

def import_many_flat_samples_add_targets(starting_number, ending_number):
    targets_df = tg.import_training_targets()
    sample_name_list = []
    target_list = []

    #print("appending targets")
    #get the list of sample names for the range
    for entry in range(starting_number,ending_number+1):
        #print("Entry : Target : ID" , str(entry), str(targets_df["target"].iloc[entry]) ,str(targets_df["id"].iloc[entry]))
        sample_name_list.append(targets_df["id"].iloc[entry])
        target_list.append(targets_df["target"].iloc[entry])

    samples_df = import_list_of_flat_training_samples(sample_name_list)
    samples_df["target"] = target_list
    samples_df["id"] = sample_name_list
    return samples_df



def import_single_samples_of_number(number):
    targets_df = tg.import_training_targets()
    return targets_df["id"].values[number]




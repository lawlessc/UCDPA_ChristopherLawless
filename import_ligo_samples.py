import math

import pandas as pd
import numpy as np
import get_ligo_path as gp
import import_targets as tg
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MaxAbsScaler,MinMaxScaler,Normalizer, normalize
import visualisers as vs
from fractions import gcd


#just a test on a single file, shows 3 rows, 4096 columns
#These rows correspond to the 3 detectors.
def import_test():
    testnpy = np.load("data/00000e74ad.npy")
    print(type(testnpy))
    print(testnpy.shape)
    return testnpy

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
   # flat = testnpy.flatten()

    #predictor_scaler = StandardScaler().fit(testnpy)
    #predictors = predictor_scaler.transform(testnpy)

    #print("shape:"+str(testnpy.shape))

    scaler = MinMaxScaler()
    # scaler.fit(testnpy)

    # predictors = [] #this is just so i don't have an empty array
    # scaler.fit(testnpy[0,:])
    # predictors[0] = scaler.transform(testnpy[0])
    #
    # scaler.fit(testnpy[1])
    # predictors[1] = scaler.transform(testnpy[1])
    #
    # scaler.fit(testnpy[2])
    # predictors[2] = scaler.transform(testnpy[2])

    #testnpy[0,:]  = normalize(testnpy[0,:])

    #predictors = normalize(testnpy,axis=0)
    #print("test1" +str(predictors) )
    #predictors = normalize(testnpy, axis=1)
    # print("test2" + str(testnpy))

    # predictors = np.multiply(np.array(testnpy[0,:]),np.array(testnpy[1,:]))
    # predictors = np.multiply(predictors, np.array(testnpy[2, :]))

    # print("predictors: " +str(predictors))
    #
    # print("Peaks1: %i troughs: %i" , str(peak_counter(testnpy[0,:])), str(trough_counter(testnpy[0,:])))
    # print("Peaks2: %i  troughs: %i" , str(peak_counter(testnpy[1, :])), str(trough_counter(testnpy[1,:])))
    # print("Peaks3: %i  troughs: %i", str(peak_counter(testnpy[2, :])), str(trough_counter(testnpy[2,:])))


    #doing this beacause i think scientifica notation was messing with everything.
    bignum  = 1000000000000000000000
    bignum2 = 10000000000000000000000*0.3

    testnpy[0,:] =   np.multiply(bignum, testnpy[0,:])
    testnpy[1, :] =  np.multiply(bignum, testnpy[1, :])
    testnpy[2, :] =  np.multiply(bignum2, testnpy[2, :])
    #
    testnpy[2, :] = np.negative(testnpy[2, :])#maybe the location of the virgo array means flipping this will help

    greatest_common = []

    for i in range(0,4096):
        x =   math.gcd(int(testnpy[0,i]),int(testnpy[1,i]))
        greatest_common.append(x)

   # testnpy = np.append(testnpy,average_array,axis=1)

    # scaler = MinMaxScaler()
    # scaler.fit(testnpy)
    # testnpy= scaler.transform(testnpy)

    diff1= np.subtract(testnpy[0,i],greatest_common)
    #diff1=

   # difference =  np.subtract(np.array(testnpy[0,:]),testnpy[1,:])

   # added = np.add(np.array(testnpy[0,:]),testnpy[1,:])

   # diff1 = np.subtract(difference,added)
    #diff1 = np.subtract(diff1,np.array(testnpy[0, :]))



   # diff1 = np.subtract(difference,np.array(testnpy[0,:]))
    #diff1 = np.subtract(diff1, np.array(testnpy[1, :]))
    #diff2 = np.subtract(diff2, np.array(testnpy[1, :]))

   # testnpy[0, :] = np.subtract(testnpy[0, :], diff1)
 #   testnpy[1, :] = np.subtract(testnpy[1, :], diff2)





    #testnpy[0, :] = np.subtract(testnpy[0, :], diff1)
    #testnpy[1, :] = np.subtract(testnpy[1, :], diff2)
    #testnpy[2, :] = np.subtract(testnpy[2, :], diff3)

     #testnpy[0, :] = np.array(diff1)
    # testnpy[1, :] = np.array(diff2)
    # testnpy[2, :] = np.array(average_array)


    predictor_scaler = Normalizer().fit(testnpy)
    testnpy = predictor_scaler.transform(testnpy)




    # print("test3" + str(testnpy))
    #
    signal_list = [diff1]
    vs.signal_plotter(self=vs ,signals_list=  signal_list)


    #new idea, get differences between all three and sum the differences
    # diff1 = np.subtract(np.array(testnpy[0,:]),np.array(testnpy[1,:]))
    # diff2 = np.subtract(np.array(testnpy[0, :]), np.array(testnpy[2, :]))
    # diff3 = np.subtract(np.array(testnpy[2, :]), np.array(testnpy[1, :]))
    #
    # predictors = np.add(diff1, diff2)
    # predictors = np.add(predictors, diff3)

   #np.d

    # predictors = predictors - np.mean(predictors)
    # predictors = predictors / np.max(predictors)
    # #
    # print("normalized" + str(predictors))

    #predictors = normalize(predictors, axis=1)
    #predictors = normalize(predictors, axis=1)

    #print("first"+str(testnpy[0,1]))

    #scaler = MaxAbsScaler()
    #scaler.fit(testnpy)
    #predictors = scaler.transform(testnpy)

    #predictors = np.log(testnpy)
    #print("log test:"+str(predictors)) #This just breaks the data by outputing NANs for negatives

    #return predictors.flatten()
    return testnpy.flatten()

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
    print(samples_df)
    return samples_df



def import_single_samples_of_number(number):
    targets_df = tg.import_training_targets()
    return targets_df["id"].values[number]




def peak_counter(an_array):
    count =0

    for i in range(1 ,len(an_array)-1):
        if an_array[i] > an_array[i-1]:
            if an_array[i] > an_array[i+1]:
                count+=1
    return count

def trough_counter(an_array):
    count =0

    for i in range(1 ,len(an_array)-1):
        if an_array[i] < an_array[i-1]:
            if an_array[i] < an_array[i+1]:
                count+=1
    return count




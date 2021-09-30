import pandas as pd
import import_ligo_samples as samples
import kaggle_download as kg
import exploratory_analysis as ex
import HyperParameterOpto as Hp
import GridSearcher as Gr
import principal_component_analysis as pca
import bandpass_filtering as bs
import entropy_checker as ec
import gc
import NeuralNetDefiner as nn


from numpy.random import seed, RandomState
seed(1)


import tensorflow as tf
from keras import backend as K




print('Made by Christopher Lawless July 2021')

# This is for downloading the ALL the files from the Kaggle competition dataset(72gb)
# This requires you have installed the Kaggle Api and have an api key setup
# kg.begin_download()


# This is some exploratory analysis, these can be commented out if you don't want visualisations loading everytime
# the program runs.
#ex.analysis_of_targets()
#ex.analysis_of_data()
#ex.analysis_of_signal()
#samples.import_test()

# For large amounts of data beyond 10,000 this can take a while to load
# If you haven't downloaded the data via Kaggle you will only have 154 samples to use starting at 0



rs = RandomState(seed=69)

# data = samples.import_positive_flat_samples_add_targets(0,19000)
# data = samples.import_negative_flat_samples_add_targets(0,8000)
# gc.collect()

# model = nn.load_model("models/transferlearner3.h5")

data = samples.import_flat_samples_add_targets(0, 42100)


data = data.append(samples.import_many_zeros(8))
data = data.append(samples.import_many_rand(8))
data = data.append(samples.import_many_ones(8))
data = data.sample(frac=1 , random_state=rs).reset_index(drop=True)

print(data)


#data_pos = samples.import_positive_flat_samples_add_targets(0,5192)

#data_neg = samples.import_negative_flat_samples_add_targets(0,6000)

#data_single_positive = samples.import_positive_flat_samples_add_targets(0,1)

#data_single_negative = samples.import_negative_flat_samples_add_targets(0,1)

# data can also be loaded in chunks and appended to the dataframe
# data = data.append(samples.import_many_flat_samples_add_targets(20000, 20060))

# This does badpass filtering.
##bs.run(data[])

#ec.try_entropy(data)

# This does gridsearch
#gs = Gr.GridSearcher()
#gs.do_search(data, describe=True)

#pcapos = pca.do_pca_of_data(data=data_pos)

#print(pcapos)

#gs.do_auto_encoder_search(pcapos,describe=True)


#gs.do_auto_encoder_search(data_neg,describe=True)
# This does my own attempt at building a hyperameter tuning.
hypo_trainer = Hp.HyperParameterOpto()
hypo_trainer.train_network(data)
# hypo_trainer.train_loaded_network(data,model)
#hypo_trainer.train_auto_encoder_network(pcapos)

# This does PCA

#print("mixed PCA")
#pca.do_pca_of_data(data=data)

#print("postive PCA")
#pca.do_pca_of_data(data=data_pos)

#print("negative PCA")
#pca.do_pca_of_data(data=data_neg)

#print("negative data with single positive appended")
#data_neg = data_neg.append(data_single_positive)
#pca.do_pca_of_data(data=data_neg)

#print("positive data with single negative appended")
#data_pos = data_pos.append(data_single_negative)
#pca.do_pca_of_data(data=data_pos)


# Testing
# df_test_data =samples.import_number_of_flat_testing_samples(4,100)
# print(df_test_data)
# neuralnet_d.make_prediction_with("29_07_2021_21_04_16.h5",df_test_data)

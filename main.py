import pandas as pd
import import_ligo_samples as samples
import kaggle_download as kg
import exploratory_analysis as ex
import HyperParameterOpto as Hp
import GridSearcher as Gr


print('Made by Christopher Lawless July 2021')

# This is for downloading the ALL the files from the Kaggle competition dataset.
# This requires you have installed the Kaggle Api and have an api key setup
# kg.begin_download()


# This is some exploratory analysis, these can be commented out if you don't want visualisations loading everytime
# the program runs.
ex.analysis_of_targets()
ex.analysis_of_data()
ex.analysis_of_signal()

# For large amounts of data beyond 10,000 this can take a while to load
# If you haven't downloaded the data via Kaggle you will only have 154 samples to use starting at 0
data = samples.import_flat_samples_add_targets(0, 150)
# data can also be loaded in chunks and appended to the dataframe
# data = data.append(samples.import_many_flat_samples_add_targets(20000, 20060))


#This does gridsearch
gs = Gr.GridSearcher()
gs.do_search(data, describe=True)

# This does my own attempt at building a hyperameter tuning.
hypo_trainer = Hp.HyperParameterOpto()
hypo_trainer.train_network(data)

# Testing
# df_test_data =samples.import_number_of_flat_testing_samples(4,100)
# print(df_test_data)
# neuralnet_d.make_prediction_with("29_07_2021_21_04_16.h5",df_test_data)

import pandas as pd
import numpy
import import_ligo_samples
import import_ligo_samples as samples
import import_targets as tg
#import kaggle_download as kg
import exploratory_analysis as ex
import HyperParameterOpto as tr
import GridSearcher as gr

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Made by Christopher Lawless July 2021')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


#tg.import_targets()


#samples.import_flat_single_sample("00000e74ad")


#testlist= ["00000e74ad","c00a5bd72f","f31a3199d0"]

#samples.import_list_of_flat_samples(testlist)

#samples.import_many_flat_samples(0, 3)




#ex.analysis_of_targets()

#neuralnet_d = nnd.NN_definer()

#training

#data = []
data = samples.import_many_flat_samples_add_targets(0,6000)

#data.append(samples.import_many_flat_samples_add_targets(20000, 30000))
#data.append(samples.import_many_flat_samples_add_targets(50000, 60000))

#data.append(samples.import_many_flat_samples_add_targets(500000, 508000))


#data =  data.append(samples.import_many_flat_samples_add_targets(300000, 301000))
#data.append(samples.import_many_flat_samples_add_targets(200000, 210000))
#data.append(samples.import_many_flat_samples_add_targets(100000, 200000))


#trainer = tr.hyper_paramenter_training()
#trainer.train_network(data)

gs = gr.GridSearcher()

gs.do_search(data)


#Testing
#df_test_data =samples.import_number_of_flat_testing_samples(4,100)
#print(df_test_data)
#neuralnet_d.make_prediction_with("29_07_2021_21_04_16.h5",df_test_data)








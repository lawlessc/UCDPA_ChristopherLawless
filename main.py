import pandas as pd
import import_ligo_samples as samples
import kaggle_download as kg
import exploratory_analysis as ex
import hyper_parameter_opto as hp
import grid_searcher as gr
import NN_definer as nnd

print(f'Made by Christopher Lawless July 2021')

#testlist= ["00000e74ad","c00a5bd72f","f31a3199d0"]
#samples.import_list_of_flat_samples(testlist)

ex.analysis_of_targets()

#neuralnet_d = nnd.NN_definer()



samples.import_test()


#For large amounts of data beyond 10,000 this can take a while to load
data = samples.import_flat_samples_add_targets(0,500)
#data can also be loaded in chunks and appended to the dataframe
#data = data.append(samples.import_many_flat_samples_add_targets(20000, 20060))

print(data)

gs = gr.grid_searcher()
gs.do_search(data, describe=True)


#test_model = neuralnet_d.create_model()


#trainer = hp.hyper_paramenter_training()
#trainer.train_network(data)

#Testing
#df_test_data =samples.import_number_of_flat_testing_samples(4,100)
#print(df_test_data)
#neuralnet_d.make_prediction_with("29_07_2021_21_04_16.h5",df_test_data)
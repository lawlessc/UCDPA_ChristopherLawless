import numpy
from keras.optimizers import SGD ,Adam, RMSprop
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler,MaxAbsScaler
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
from tensorflow.keras import mixed_precision

import NN_definer as nnd

class grid_searcher:
    '''This is a class for doing'''


    neuralnet_d = nnd.NN_definer()

    def do_search(self,data , describe=False):
        '''This takes a  single dataframe as input and does a gridsearch , you can also set it to print information on the dataframe'''

        if describe == True:
            #print(data.describe())
            print(data.head())

        tf.keras.backend.set_floatx("float64") # I did this because the Ligo Data are 64 bit floats.
        print("keras data type" + tf.keras.backend.floatx())

        tf.device("cpu:0")#I added this try out and found it faster than overpriced GPU

        #policy = mixed_precision.Policy('mixed_float16')
        #mixed_precision.set_global_policy(policy)

        data_ = data.drop(["target"], axis=1)
        data_= data_.drop(["id"], axis=1)

        predictors = data_.to_numpy()

        #print("grid")
        #print(str(predictors.shape[0]))
        #print(str(predictors.shape[1]))

      #  print("grid")
       # print(predictors))




        #print("value func"+str( data_.values))
        #print("to_numpy" + str(predictors))



        model = KerasClassifier(build_fn=self.neuralnet_d.create_model)

        param_grid =  {"epochs": [100,50],
                      "first_layer": [10,20,100],
                      "hidden_layers": [1],
                      "layer_widths": [30],
                      "optimizer": ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
                      "winit": ["glorot_normal"],
                      "batch_size": [32],
                      "dropout": [0.01],
                      "decay": [0.01]
                      }

        grid = GridSearchCV(estimator=model,param_grid= param_grid, n_jobs= 1, cv=2,verbose=3)

        #print(data.target.values)
        #targets = np.flip(data.target.values)
        #print(targets)

        # detector_readings[2, :] = np.flip(detector_readings[2, :])


        #targets = to_categorical(data.target)

        X_train, X_test, y_train, y_test =train_test_split(predictors, data.target.to_numpy() ,shuffle=False)#test sets are empty because i am testing in gridsearch

        predictor_scaler = MinMaxScaler().fit(X_train)
        X_train = predictor_scaler.transform(X_train)

        results = grid.fit(X_train,y_train)


        print("Best Score: %f " ,results.best_score_)
        print("Best Parameters: %s" , results.best_params_)




    def do_ensemble_search(self, data):
        '''This is incomplete but is to perform a grid search for training ensemble networks'''
        predictors = data.drop(["target"], axis=1).to_numpy()
        # predictor_scaler = StandardScaler().fit(predictors)
        # predictors = predictor_scaler.transform(predictors)
        # predictors.reshape(2,3,2)

        model = KerasClassifier(build_fn=self.neuralnet_d.create_model,)

        param_grid = {"epochs": [1,2,3,4,5,10,11,100],
                      "first_layer": [10,200],
                      "hidden_layers": [1, 2],
                      "layer_widths": [10],
                      "optimizer": ["adam"],
                      "winit": ["normal"],
                      "batch_size": [128],
                      "dropout": [0, 0.1, 0.3],
                      "decay": [0.01]
                      }

        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2, verbose=3)

        results = grid.fit(predictors, data.target.values)

        print("Best Score: %f ", results.best_score_)
        print("Best Parameters: %s", results.best_params_)
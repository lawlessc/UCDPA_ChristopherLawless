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


import NN_definer as nnd

class GridSearcher:


    neuralnet_d = nnd.NN_definer()

    def do_search(self,data):
        #print("keras type" +tf.keras.backend.floatx())
        tf.keras.backend.set_floatx("float64") # I did this because the Ligo Data
        #print("keras type2" + tf.keras.backend.floatx())


        data_ = data.drop(["target"], axis=1)
        predictors = data_.drop(["id"], axis=1).to_numpy()

       # reshaped = predictors.reshape((503808,3,1))

        #print("predictor 0_0"+str(predictors[0,0]))
        #print("predictor 0_0" + str(predictors[0, 1]))
        #print("predictor 0_0" + str(predictors[1, 0]))
        #print("predictor" + str(predictors[1, 12287]))




        #predictor_scaler = StandardScaler().fit(predictors)
        #predictors = predictor_scaler.transform(predictors)

        #print("Type" + str(type(predictors)))

        #predictors_ = np.ndarray((len(data)),np.float64)

        #for tup in predictors:
        #    predictors_ = tup[0,12287]



        #predictor_scaler = MinMaxScaler()
        #predictor_scaler.fit(predictors)
        #predictors = predictor_scaler.transform(predictors)

        #predictor_scaler = MaxAbsScaler().fit(predictors)
        #predictors = predictor_scaler.transform(predictors)

        #pca = PCA(n_components=50)

        #predictors_pca = []

        #for item in predictors:
         #   x = numpy.empty()
         #   x


          #  y =pca.fit_transform(item)



        #robusts = RobustScaler()
        #predictors = robusts.fit_transform(predictors)



        #small_adam = Adam(learning_rate=0.000001)
        #big_adam = Adam(learning_rate=0.001)

        model = KerasClassifier(build_fn=self.neuralnet_d.create_model)

        param_grid =  {"epochs": [50,75],
                      "first_layer": [3,4,5],
                      "hidden_layers": [1,2],
                      "layer_widths": [3,4],
                      "optimizer": ["sgd"],
                      "winit": ["glorot_normal"],
                      "batch_size": [128,1000],
                      "dropout": [0],
                      "decay": [0.01]
                      }

        grid = GridSearchCV(estimator=model,param_grid= param_grid, n_jobs= 1, cv=2,verbose=3)

        print("lenght:"+str(len(data.target.values)))

        print("lenght:" + str(len(data.target.values)))



        #targets = to_categorical(data.target.values)

        X_train, X_test, y_train, y_test =train_test_split(predictors, data.target.values, test_size = 0.01, random_state = 42)

        print("X_train:" + str(len(X_train)))
        print("X_test:" + str(len(X_test)))
        print("y_train:" + str(len(y_train)))
        print("y_test:" + str(len(y_test)))

        results = grid.fit(X_train,y_train)
        #results = grid.fit(predictors_, None)

        print("Best Score: %f " ,results.best_score_)
        print("Best Parameters: %s" , results.best_params_)

        #grid.e




    def do_ensemble_search(self, data):
        predictors = data.drop(["target"], axis=1).to_numpy()
        # predictor_scaler = StandardScaler().fit(predictors)
        # predictors = predictor_scaler.transform(predictors)
        # predictors.reshape(2,3,2)

        model = KerasClassifier(build_fn=self.neuralnet_d.create_model)

        param_grid = {"epochs": [1,2,3,4,5,10,11,100],
                      "first_layer": [10,200],
                      "hidden_layers": [1, 2],
                      "layer_widths": [10],
                      "optimizer": ["adam"],
                      "winit": ["normal"],
                      "batch_size": [2,10,100,210],
                      "dropout": [0, 0.1, 0.3],
                      "decay": [0.01]
                      }

        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2, verbose=3)

        results = grid.fit(predictors, data.target.values)

        print("Best Score: %f ", results.best_score_)
        print("Best Parameters: %s", results.best_params_)
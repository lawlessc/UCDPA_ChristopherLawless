import numpy
from keras.optimizers import SGD, Adam, RMSprop
import keras.utils as ku
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
from tensorflow.keras import mixed_precision

import NeuralNetDefiner as Nnd


class GridSearcher:
    """This is a class for doing gird searches using GridSearchCV"""

    neuralnet_d = Nnd.NeuralNetDefiner()

    def do_search(self, data, describe=False):
        """This takes a  single dataframe as input and does a gridsearch , you can also set it to print information
        on the dataframe """
        if describe == True:
            # print(data.describe())
            print(data.head())

        # These remove the Target and ID columns for the dataframe.
        data_ = data.drop(["target"], axis=1)
        data_ = data_.drop(["id"], axis=1)
        predictors = data_.to_numpy()

        tf.keras.backend.set_floatx("float32")  # I did this because the Ligo Data are 64 bit floats.
        print("keras data type" + tf.keras.backend.floatx())
        # tf.device("cpu:0")#I added this try rule out GPU issues

        model = KerasClassifier(build_fn=self.neuralnet_d.create_model)
        opto_ssg1 = SGD(learning_rate=0.000001)
        opto_ssg2 = SGD(learning_rate=0.0000001)
        opto_ssg3 = SGD(learning_rate=0.00000001)

        param_grid = {"epochs": [25],
                      "first_layer": [75],
                      "hidden_layers": [10, 11, 12],
                      "layer_widths": [10],
                      "cnn_window_size": [32],
                      "max_pool_size": [10, 20, 30],
                      "optimizer": [opto_ssg2],
                      "winit": ["random_uniform"],
                      "batch_size": [128],
                      "dropout": [0.01],
                      "decay": [0.01]
                      }

        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2, verbose=3)

        targets = to_categorical(data.target.to_numpy())
        X_train, X_test, y_train, y_test = train_test_split(predictors, targets,
                                                            shuffle=False)  # test sets are empty because i am testing with cross validation in gridsearch

        predictor_scaler = MinMaxScaler().fit(X_train)
        X_train = predictor_scaler.transform(X_train)

        results = grid.fit(X_train, y_train)
        print("Best Score: %f ", results.best_score_)
        print("Best Parameters: %s", results.best_params_)

        # I am printing these because Gridsearch does not output their instance names
        # but only outputs their memory location e.g "0x000001FBDE926B70"
        print("Optimizer1:" + str(opto_ssg1))
        print("Optimizer2:" + str(opto_ssg2))
        print("Optimizer3:" + str(opto_ssg3))

    def do_ensemble_search(self, data):
        """This is incomplete but is to perform a grid search for training ensemble networks, it takes in a
        pandaframe of the training data """
        predictors = data.drop(["target"], axis=1).to_numpy()
        model = KerasClassifier(build_fn=self.neuralnet_d.create_model, )
        param_grid = {"epochs": [1, 2, 3, 4, 5, 10, 11, 100],
                      "first_layer": [10, 200],
                      "hidden_layers": [1],
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

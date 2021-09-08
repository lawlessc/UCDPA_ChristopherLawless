import numpy
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.optimizers.schedules import CosineDecay, LearningRateSchedule
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
from datetime import datetime

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

        predictors = data.drop(["id", "target"], axis=1)

        predictors = predictors.to_numpy()

        tf.keras.backend.set_floatx("float32")  # I did this because the Ligo Data are 64 bit floats.
        print("keras data type" + tf.keras.backend.floatx())
        # tf.device("cpu:0")#I added this try rule out GPU issues

        model = KerasClassifier(build_fn=self.neuralnet_d.create_model)
        # opto_ssg1 = SGD(learning_rate=0.000001)

        # decay_steps = 1000
        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.999, decay_steps=1000)

        opto_ssg2 = SGD(learning_rate=0.0000001, )
        opto_ssg3 = SGD(learning_rate=lr_decayed_fn)

        early_stopping = EarlyStopping(monitor='loss', patience=3)

        param_grid = {"epochs": [300],
                      "first_layer": [10],
                      "hidden_layers": [4],
                      "layer_widths": [12],
                      "cnn_window_size": [31],
                      "max_pool_size": [10],
                      "optimizer": [opto_ssg3],
                      "winit": ["random_uniform"],
                      "batch_size": [512],
                      "dropout": [0.70],
                      "decay": [0.01],
                      "lossf": ["mean_squared_error"]
                     # "callbacks": [early_stopping]
                      }

        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2, verbose=3)

        # targets = to_categorical(data.target.to_numpy())

        np.set_printoptions(threshold=np.inf)

        # print("predictors" + str(predictors))
        # print("---------------------------------------------------------------")
        # print("targets" + str(targets))

        X_train, X_test, y_train, y_test = train_test_split(predictors, data.target.to_numpy(),
                                                            shuffle=False)  # test sets are empty
        # because i am testing with cross validation in gridsearch

        # predictor_scaler = MinMaxScaler().fit(X_train)
        # X_train = predictor_scaler.transform(X_train)

        predictor_scaler = StandardScaler().fit(X_train)
        # StandardScaler.set_params(with_mean=False)
        X_train = predictor_scaler.transform(X_train)

        results = grid.fit(X_train, y_train)
        # results.
        print("Best Score: %f ", results.best_score_)
        print("Best Parameters: %s", results.best_params_)
        #
        # I am printing these because Gridsearch does not output their instance names
        # but only outputs their memory location e.g "0x000001FBDE926B70"
        # print("Optimizer1:" + str(opto_ssg1))
        print("Optimizer2:" + str(opto_ssg2))
        print("Optimizer3:" + str(opto_ssg3))

    def do_auto_encoder_search(self, data, describe=False):
        """This takes a  single dataframe as input and does a gridsearch , you can also set it to print information
        on the dataframe """
        if describe == True:
            # print(data.describe())
            try:
                print(data.head())
            except:
                print("")

        # These remove the Target and ID columns for the dataframe.
        try:
            data = data.drop(["target", "id"], axis=1)

        except:
            print("")

        try:
            predictors = data.to_numpy()
        except:
            predictors = data

        tf.keras.backend.set_floatx("float32")  # I did this because the Ligo Data are 64 bit floats.
        # gpu_options = tf.GPUOptions(allow_growth=True)
        # session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        print("keras data type" + tf.keras.backend.floatx())
        tf.device("cpu:0")  # I added this try rule out GPU issues

        model = KerasClassifier(build_fn=self.neuralnet_d.create_autoencoder)

        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.9, decay_steps=2000)

        opto_ssg2 = SGD(learning_rate=0.0000001, )
        opto_ssg3 = SGD(learning_rate=lr_decayed_fn)
        adam2 = Adam(learning_rate=lr_decayed_fn)

        early_stopping = EarlyStopping(monitor='loss', patience=3)

        current = datetime.now()
        date = current.strftime("models/%d_%m_%Y_%H_%M_%S")


        model_checkpoint= ModelCheckpoint(date+".h5",monitor='accuracy',save_best_only=True)

        param_grid = {"epochs": [10],
                      "first_layer": [100],
                      "hidden_layers": [2],
                      "layer_widths": [40],
                      "cnn_window_size": [100],
                      "max_pool_size": [1000],
                      "optimizer": [Adam(lr=0.0005)],
                      "winit": ["random_uniform","gl_uniform"],
                      "batch_size": [512],
                      "dropout": [0.20],
                      "lossf": ["mean_squared_error"],#"kl_divergence","binary_crossentropy",

                     # ['mean_squared_error', "binary_crossentropy",
                      # "categorical_crossentropy", "kl_divergence", "poisson"],
                      #"callbacks": [model_checkpoint],
                      "decay": [0.01]

                      }

        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2, verbose=3)


        # targets = to_categorical(data.target.to_numpy())

        #np.set_printoptions(threshold=np.inf)


        # X_train, X_test, y_train, y_test = train_test_split(predictors, predictors,shuffle=False)  # test sets are empty
        # because i am testing with cross validation in gridsearch

        #predictor_scaler = MinMaxScaler().fit(predictors)
        #predictors = predictor_scaler.transform(predictors)

        predictor_scaler = MinMaxScaler(feature_range=(0, 1)).fit(predictors)
        predictors = predictor_scaler.transform(predictors)

        # predictor_scaler = StandardScaler().fit(predictors)
        # predictors = predictor_scaler.transform(predictors)


        print("predictors")
        print(predictors)
        #predictor_scaler = StandardScaler().fit(predictors)
        #predictors = predictor_scaler.transform(predictors)

        results = grid.fit(predictors, predictors)
        # results.
        #self.neuralnet_d.save_model(grid.best_estimator_) # We attempt to save the model at the end of gridsearch

        #results.best_estimator_.
        print("Best Score:", results.best_score_)
        print("Best Parameters:", results.best_params_)
        #
        # I am printing these because Gridsearch does not output their instance names
        # but only outputs their memory location e.g "0x000001FBDE926B70"
        # print("Optimizer1:" + str(opto_ssg1))
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

        self.neuralnet_d.save_model(grid.best_estimator_)# We attempt to save the model at the end of gridsearch
        print("Best Score: %f ", results.best_score_)
        print("Best Parameters: %s", results.best_params_)

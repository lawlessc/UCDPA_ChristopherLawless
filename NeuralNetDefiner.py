import keras
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, Dropout,Reshape, Conv1D, MaxPooling1D,Rescaling, Conv2D ,AveragePooling1D
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.models import load_model, Sequential, Layer
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers
import visualisers as vs


# Made by Christopher Lawless July 2021


class NeuralNetDefiner:
    """This is a class for creating and saving various types of model, for Gridsearch use the create_model function"""

    def load_model(self, model_name):
        """This takes a single string of a model and loads it from the models folder"""
        return load_model(model_name)

    def create_model(self, first_layer=30, hidden_layers=1, max_pool_size=10, layer_widths=10, optimizer="sgd",
                     winit="glorot_uniform", cnn_window_size=30, dropout=0.01, decay=0.01, input_shape=(12288,),
                     lossf='mean_squared_error' ):
        """function tales in the first layer width, dropout, decay, number or hidden layers and their width and the
        optimizer and initial weights this is mainly for use with GridSearchCV but can be used discretely if you just
        want a model """
        #model = Sequential()
        # Input Layer
        #model.add(Reshape((4096, 3), input_shape=(12288,)))

        inputs = keras.Input(shape=(12288,1))

       # reshaped = layers.Reshape((4096, 3), input_shape=(12288,))(inputs)
        #reshaped=keras.Input(shape=(12288,))

        #densex = layers.Dense(10, activation="relu")
        densex = Conv1D(16, 100, padding='same', activation='relu')(inputs)
        x = AveragePooling1D(32, strides=20, padding='same')(densex)
        densex2 = Conv1D(16, 100, padding='same', activation='relu')(x)
        x2 = AveragePooling1D(32, strides=20, padding='same')(densex2)


        densey = Conv1D(16, 10, padding='same', activation='relu')(inputs)
        y = AveragePooling1D(32, strides=2, padding='same')(densey)
        densey2 = Conv1D(16, 10, padding='same', activation='relu')(y)
        y2 = AveragePooling1D(32, strides=2, padding='same')(densey2)

        densez = Conv1D(16, 2048, padding='same', activation='relu')(inputs)
        z = AveragePooling1D(32, strides=2048, padding='same')(densez)
        densez2 = Conv1D(16, 2048, padding='same', activation='relu')(z)
        z2 = AveragePooling1D(32, strides=2048, padding='same')(densez2)

        lazer = LeakyReLU(alpha=0.5)(inputs)
        denzer = Dense(16, activation="relu")(lazer)

        merged = keras.layers.concatenate([x2, y2, z2,denzer], axis=1)
        lazy =   LeakyReLU(alpha=0.3)(merged)
        lazy2 = LeakyReLU(alpha=0.3)(lazy)
        dens = Dense(10, activation="relu")(lazy2)
        lazy3 = LeakyReLU(alpha=0.3)(dens)
        dens2 = Dense(5, activation="relu")(lazy3)

        #merged = Flatten()(merged)

        outputs = layers.Dense(1,activation="sigmoid")(dens2)
        #outputs = outputs(outputs)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # model.add(Dense(100, activation='relu', kernel_initializer='he_uniform',input_shape=(4096,3)))
        # model.add(Conv1D(first_layer, kernel_size=(3), activation='relu', input_shape=(4096, 3),
        #                  kernel_initializer="he_uniform"))
        # model.add(LeakyReLU(alpha=0.3))
        # model.add(LeakyReLU(alpha=0.3))
        # model.add(LeakyReLU(alpha=0.3))
        # #model.add(Dense(1, activation="sigmoid"))
        # model.add(LeakyReLU(0.001))
        #model.add(Dense(1, activation="sigmoid"))

        # Compile
        model.compile(optimizer=optimizer, loss=lossf, metrics=["accuracy"])
        return model

    def create_autoencoder(self, first_layer=30, hidden_layers=1, max_pool_size=10, layer_widths=10, optimizer="adam",
                     winit="random_uniform", cnn_window_size=30, dropout=0.01, decay=0.01, input_shape=(12288,),
                           lossf='mean_squared_error'
                           ):
        """function tales in the first layer width, dropout, decay, number or hidden layers and their width and the
        optimizer and initial weights this is mainly for use with GridSearchCV but can be used discretely if you just
        want a model """
        model = Sequential()


       #model.add(Reshape((4096, 3), input_shape=(12288,)))
        model.add(Dense(55, activation="sigmoid", kernel_initializer=winit,input_shape=(55,)))
      #  model.add(LeakyReLU(55, input_shape=(55,)))
        #model.add(Dropout(dropout))


       # model.add(Dense(40, activation="sigmoid", kernel_initializer=winit))
        for x in range(hidden_layers):
            model.add(LeakyReLU(alpha=0.3))
        #model.add(LeakyReLU(layer_widths))
        #model.add(Dense(layer_widths, activation="tanh", kernel_initializer=winit))
        #model.add(LeakyReLU(55))


        model.add(Dense(55, activation="sigmoid",kernel_initializer=winit))

        # Compile
        model.compile(optimizer=optimizer, loss=lossf ,metrics=["accuracy"])
        return model




    def create_ensemble_model(self, first_layer, dropout, decay, hidden_layers, layer_widths, optimizer, winit):
        """This is not in use but was intended to be for training a model, that learns the outputs of other models"""
        model = Sequential()
        model.add(LeakyReLU(first_layer, input_shape=(12288 + 5,)))
        for x in range(hidden_layers):
            model.add(LeakyReLU(layer_widths))
        model.add(Dense(2, activation="softmax"))
        # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
        return model

    def define_model(self, data, hidden_layers, hidden_layer_width):
        """This is not in use but was intended for training models with the hyper_parameter_opto.py class"""
        model = Sequential()
        model.add(LeakyReLU(hidden_layer_width, input_shape=(data.shape[1] - 2,)))
        # model.add(Dense(hidden_layer_width, activation="relu",input_shape=(data.shape[1] - 1,)))
        # model.add(SimpleRNN(units=32,input_shape=(4096,3) ,activation="relu"))
        # model.add(LSTM(100))
        # model.add(layers.MaxPooling1D((12288, 12288)))#lets just add a pooling layer and see what happens
        # model.add(LeakyReLU(4))
        # model.add(LeakyReLU(60, input_shape=(data.shape[1] - 1,)))
        # model.add(Dropout(0.1))
        # model.add(Dense(2, activation="relu", kernel_constraint=maxnorm(3)))
        for x in range(hidden_layers):
            print("layer added")
            model.add(LeakyReLU(hidden_layer_width))
        # model.add(Dense(3, activation="sigmoid"))
        # model.add(Dense(3, activation="relu", kernel_constraint=maxnorm(3)))
        # model.add(Dense(42, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        print("compile Model")
        return model

    def compile_model(self, model, optimizer):
        """This compiles a model , it takes the model and an optimizer"""
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
        return model




    def fit_model_auto_encoder(self, data, epochs, model, batch_size,use_early_stopping_time=0):
        """Takes in data, epochs, a keras model and batch_size, and if early stopping is to be used."""
        print("fit model")

        predictors = data

        predictor_scaler = MinMaxScaler(feature_range=(0,1)).fit(predictors)
        predictors = predictor_scaler.transform(predictors)

        #predictor_scaler = StandardScaler().fit(predictors)
        #predictors = predictor_scaler.transform(predictors)


       # print(data.target.values)
        print(predictors[1])

        #I create a callback list for potential callbacks i might want.
        callbacks =[]

        #This inserts a an early stopping monitor into callbacks with a set time.
        if use_early_stopping_time > 0:
            callbacks = [EarlyStopping(patience=use_early_stopping_time, monitor="val_accuracy")]


        mt = model.fit(predictors, predictors, epochs=epochs, batch_size=batch_size,
                       validation_split=0.15, callbacks=callbacks)  #

        mlist = [mt]
        return mt, model






    def fit_model(self, data, epochs, model, batch_size,use_early_stopping_time=0):
        """Takes in data, epochs, a keras model and batch_size, and if early stopping is to be used."""
        print("fit model")
        predictors = data.drop(["id"], axis=1)
        predictors = predictors.drop(["target"], axis=1).to_numpy()  # .as_matrix()
        # targets = to_categorical(data.target)

        predictor_scaler = StandardScaler().fit(predictors)
        predictors = predictor_scaler.transform(predictors)

        print(data.target.values)
        print(predictors[1])

        #I create a callback list for potential callbacks i might want.
        callbacks =[]

        #This inserts a an early stopping monitor into callbacks with a set time.
        if use_early_stopping_time > 0:
            callbacks = [EarlyStopping(patience=use_early_stopping_time, monitor="val_accuracy")]


        mt = model.fit(predictors, data.target.to_numpy(), epochs=epochs, batch_size=batch_size,
                       validation_split=0.25, callbacks=callbacks)  #

        mlist = [mt]
        return mt, model

    def save_model(self, model):
        """This saves a model to the models folder with date and time as the filename"""
        current = datetime.now()
        date = current.strftime("models/%d_%m_%Y_%H_%M_%S")
        model.save(date + ".h5")

    def trainer_loop(self, predictors, target, model):
        """This is just a loop for training a model multiple times, it takes predictors,targets and the model object
        as input """
        for lr in self.learning_rate:
            # self.model = get_new_model()
            # optimizer = SGD(lr=lr)
            # self.model.compile(optimizer=optimizer, loss = "categorocal_crossentropy")
            model.fit(predictors, target)

    def verify_model_info(self, model):
        '''This just prints out a models loss and a summary of the model'''
        print("Loss: " + self.model.loss)
        model.summary()

    def make_single_prediction_with(self,model_name,data_to_predict_with):
        loaded_model = load_model(model_name)
        predictions = pd.DataFrame()
        probability_true =[]

        for index ,row in data_to_predict_with.iterrows():
            #print(row)
            predictions = loaded_model.predict((pd.DataFrame(row).T).values)
            print(pd.DataFrame(row).T)
            probability_true.append(predictions[:,1])

        for prediction in probability_true:
            print(prediction)

    def make_single_prediction_with_model(self, model, data_to_predict_with):
       # loaded_model = load_model(model_name)
        predictions = pd.DataFrame()
        probability_true = []

        for index, row in data_to_predict_with.iterrows():
            predictions = model.predict((pd.DataFrame(row).T).values)
            print(pd.DataFrame(row).T)
            probability_true.append(predictions[:, 1])

        for prediction in probability_true:
            print(prediction)




    def retrain_model(self,model_name,data):
        loaded_model = load_model(model_name)

        predictors = data.drop(["target"], axis=1).values  # .as_matrix()
        targets = to_categorical(data.target)

        print(data.target.values)
        print(predictors)

        mt = self.model.fit(predictors, data.target.values, epochs=1000, validation_split=0.10)  # ,use_multiprocessing=True) This seems to be only for training large pools of models

        # early_stopping_monitor= EarlyStopping(patience=2)
        # self.model.fit(predictors,target,validation_split = 0.3,nb_epoch=20
        # ,callbacks=[early_stopping_monitor])
        mlist = [mt]
        vs.validation_plot(self=vs, model_list=mlist)
        vs.accuracy_plot(self=vs, model_list=mlist)
        self.save_model()

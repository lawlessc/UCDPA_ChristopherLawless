import keras
import pandas as pd
from keras.layers import Dense , Dropout,RNN, SimpleRNN,GRU,LSTM ,Reshape ,Conv1D,Conv2D, MaxPooling1D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU ,PReLU,ELU
from keras.utils.np_utils import to_categorical
from keras.models import load_model ,Sequential
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
#from  tf.cpm

#tf.compat.v1.spectral

from tensorflow.keras import mixed_precision
from keras import layers

#Made by Christopher Lawless July 2021

from datetime import datetime
import visualisers as vs

class NN_definer:

    def load_model(self, model_name):
        '''This takes a single string of a model and loads it from the models folder'''
        return load_model(model_name)

    def create_model(self, first_layer, dropout, decay, hidden_layers, layer_widths, optimizer, winit):
        '''function tales in the first layer width, dropout, decay, number or hidden layers and their width and the optimizer and initial weights
        this is mainly for use with GridSearchCV but can be used discretely'''
        model = Sequential()
        model.add(Reshape((12288, 1), input_shape=(12288,)))
        model.add(Conv1D(first_layer,10, activation='relu', input_shape=(122881, 1),kernel_initializer=winit))
        #model.add(Conv1D(100, 10, activation='relu'))
        #model.add(Conv1D(30, 10, activation='relu'))
        #model.add(Dropout(dropout))
        model.add(MaxPooling1D(10))
        #model.add(Conv1D(50, 5, strides=5, activation='relu'))
        #model.add(MaxPooling1D(5))
        #model.add(Conv1D(110, 1024, strides=1024, activation='relu', input_shape=(4096, 1)))
        #model.add(LeakyReLU(first_layer, input_shape=(12288,)))
        #model.add(Dropout(dropout))
        #model.add(SimpleRNN(units=first_layer, input_shape=(3, 4096), activation="relu", return_sequences=False,kernel_initializer=winit))
                           # kernel_regularizer=l2(decay), recurrent_regularizer=l2(decay), bias_regularizer=l2(decay)))
        for x in range(hidden_layers):
            model.add(LeakyReLU(layer_widths))
            #model.add(Dense(int(layer_widths),activation="relu", kernel_initializer=winit))

        model.add(Dense(1, activation="sigmoid"))


        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
        #model.summary()
        return model







    # def create_ensemble_model(self, first_layer, dropout, decay, hidden_layers, layer_widths, optimizer, winit):
    #     model = Sequential()
    #     # model.add(Reshape((4096, 3), input_shape=(12288,)))
    #     #if dropout > 0:
    #         #model.add(Dropout(dropout, input_shape=(12288,)))  # This might overkill considering the dataset is already full of noise
    #     model.add(LeakyReLU(first_layer, input_shape=(12288+5,)))
    #     # model.add(Dropout(dropout))
    #     for x in range(hidden_layers):
    #         model.add(LeakyReLU(layer_widths))
    #
    #     #model.add(LeakyReLU(4))
    #     #model.add(Dense(2, activation="sigmoid"))
    #     model.add(Dense(2, activation="softmax"))
    #     # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
    #     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
    #     return model
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # def define_model(self, data, hidden_layers, hidden_layer_width):
    #     model = Sequential()
    #
    #     #model.add(Dropout(0.03,  input_shape=(data.shape[1] - 1,))) #This might overkill considering the dataset is already full of noise
    #     model.add(LeakyReLU(hidden_layer_width,input_shape=(data.shape[1] - 1,)))
    #     #model.add(Dense(hidden_layer_width, activation="relu",input_shape=(data.shape[1] - 1,)))
    #     #model.add(SimpleRNN(units=32,input_shape=(4096,3) ,activation="relu"))
    #     #model.add(LSTM(100))
    #     #model.add(layers.MaxPooling1D((12288, 12288)))#lets just add a pooling layer and see what happens
    #     #model.add(LeakyReLU(4))
    #     #model.add(LeakyReLU(60, input_shape=(data.shape[1] - 1,)))
    #     #model.add(Dropout(0.1))
    #    #model.add(Dense(2, activation="relu", kernel_constraint=maxnorm(3)))
    #     for x in range(hidden_layers):
    #         print("layer added")
    #         model.add(LeakyReLU(hidden_layer_width))
    #         #model.add(LSTM(hidden_layer_width))
    #         #model.add(Dense(hidden_layer_width, activation="relu"))
    #     #model.add(Dense(3, activation="sigmoid"))
    #     #model.add(Dense(3, activation="relu", kernel_constraint=maxnorm(3)))
    #     #model.add(Dense(42, activation="relu"))
    #     model.add(Dense(1, activation="sigmoid"))
    #     print("compile Model")
    #     return model

    # def compile_model(self,model ,optimizer):
    #     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
    #     return model
    #
    #
    # def fit_model(self,data,epochs ,model,epoch_batch_size):
    #     print("fit model")
    #     predictors = data.drop(["target"],axis=1).to_numpy()#.as_matrix()
    #     #targets = to_categorical(data.target)
    #
    #     predictor_scaler= StandardScaler().fit(predictors)
    #
    #     predictors= predictor_scaler.transform(predictors)
    #
    #     print(data.target.values)
    #     print(predictors[1])
    #
    #     early_stopping_monitor= EarlyStopping(patience=14,monitor="val_accuracy")
    #
    #     mt =model.fit(predictors,data.target.to_numpy(), epochs=epochs,batch_size=epoch_batch_size ,
    #                   validation_split = 0.25,callbacks=[early_stopping_monitor])#
    #
    #     mlist = [mt]
    #     #vs.validation_plot(self=vs,model_list=mlist)
    #     #vs.accuracy_plot(self=vs, model_list=mlist)
    #     #self.save_model(model)
    #     return mt,model


    def save_model(self,model):
        current = datetime.now()
        date = current.strftime("models/%d_%m_%Y_%H_%M_%S")
        model.save(date+".h5")


    def trainer_loop(self, predictors, target,model):
       for lr in self.learning_rate:
           #self.model = get_new_model()
           #optimizer = SGD(lr=lr)
           #self.model.compile(optimizer=optimizer, loss = "categorocal_crossentropy")
           model.fit(predictors,target)

    def verify_model_info(self,model):
        print("Loss: " + self.model.loss)
        model.summary()



    # def make_single_prediction_with(self,model_name,data_to_predict_with):
    #     loaded_model = load_model(model_name)
    #     predictions = pd.DataFrame()
    #     probability_true =[]
    #
    #     for index ,row in data_to_predict_with.iterrows():
    #         #print(row)
    #         predictions = loaded_model.predict((pd.DataFrame(row).T).values)
    #         print(pd.DataFrame(row).T)
    #         probability_true.append(predictions[:,1])
    #
    #     for prediction in probability_true:
    #         print(prediction)
    #
    # def make_single_prediction_with_model(self, model, data_to_predict_with):
    #    # loaded_model = load_model(model_name)
    #     predictions = pd.DataFrame()
    #     probability_true = []
    #
    #     for index, row in data_to_predict_with.iterrows():
    #         predictions = model.predict((pd.DataFrame(row).T).values)
    #         print(pd.DataFrame(row).T)
    #         probability_true.append(predictions[:, 1])
    #
    #     for prediction in probability_true:
    #         print(prediction)
    #
    #
    #
    #
    # def retrain_model(self,model_name,data):
    #     loaded_model = load_model(model_name)
    #
    #     predictors = data.drop(["target"], axis=1).values  # .as_matrix()
    #     targets = to_categorical(data.target)
    #
    #     print(data.target.values)
    #     print(predictors)
    #
    #     mt = self.model.fit(predictors, data.target.values, epochs=1000,
    #                         validation_split=0.10)  # , use_multiprocessing=True) This seems to be only for training large pools of models
    #
    #     # early_stopping_monitor= EarlyStopping(patience=2)
    #     # self.model.fit(predictors,target,validation_split = 0.3,nb_epoch=20
    #     # ,callbacks=[early_stopping_monitor])
    #     mlist = [mt]
    #     vs.validation_plot(self=vs, model_list=mlist)
    #     vs.accuracy_plot(self=vs, model_list=mlist)
    #     self.save_model()
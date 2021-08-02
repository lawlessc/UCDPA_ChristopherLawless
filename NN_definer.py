import keras
import pandas as pd
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU ,PReLU,ELU
from keras.utils.np_utils import to_categorical
from keras.models import load_model ,Sequential
from keras.optimizers import SGD ,Adam
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import Conv2D
#Made by Christopher Lawless July 2021

import matplotlib.pyplot as plt
from datetime import datetime
import visualisers as vs

class NN_definer:

    def load_model(self, model_name):
        return load_model(model_name)

    def define_model(self, data, hidden_layers, hidden_layer_width):
        model = Sequential()

        print("Specify Model")

        #model.add(Dropout(0.03,  input_shape=(data.shape[1] - 1,))) #This might overkill considering the dataset is already full of noise

        #model.add(LeakyReLU(hidden_layer_width,input_shape=(data.shape[1] - 1,)))
        model.add(Dense(hidden_layer_width, activation="relu",input_shape=(data.shape[1] - 1,)))
        #model.add(LeakyReLU(4))
        #model.add(LeakyReLU(60, input_shape=(data.shape[1] - 1,)))
        #model.add(Dropout(0.1))
       #model.add(Dense(2, activation="relu", kernel_constraint=maxnorm(3)))
        for x in range(hidden_layers):
            print("layer added")
            model.add(LeakyReLU(hidden_layer_width))
            #model.add(Dense(hidden_layer_width, activation="relu"))


        #model.add(Dense(3, activation="sigmoid"))
        #model.add(Dense(3, activation="relu", kernel_constraint=maxnorm(3)))
        #model.add(Dense(42, activation="relu"))

        model.add(Dense(1, activation="sigmoid"))

        print("compile Model")

        return model

    def create_model(self, hidden_layers, layer_widths, optimizer, winit):
        model = Sequential()

        #print("Specify Model")

        # model.add(Dropout(0.03,  input_shape=(data.shape[1] - 1,))) #This might overkill considering the dataset is already full of noise

        model.add(LeakyReLU(layer_widths,input_shape=(12288,)))
        #model.add(Dense(layer_widths, activation="relu", input_shape=(12288,) ,kernel_initializer=winit))
        #model.add(Dense(layer_widths, activation="sigmoid",input_shape=(12288,) ,kernel_initializer=winit))

        # model.add(LeakyReLU(4))
        # model.add(LeakyReLU(60, input_shape=(data.shape[1] - 1,)))
        # model.add(Dropout(0.1))
        # model.add(Dense(2, activation="relu", kernel_constraint=maxnorm(3)))
        for x in range(hidden_layers):
            print("layer added")
            model.add(LeakyReLU(layer_widths))
            # model.add(Dense(hidden_layer_width, activation="relu"))
            #model.add(Dense(layer_widths, activation="sigmoid"))

        # model.add(Dense(3, activation="sigmoid"))
        # model.add(Dense(3, activation="relu", kernel_constraint=maxnorm(3)))
        # model.add(Dense(42, activation="relu"))

        model.add(Dense(1, activation="sigmoid"))

        #print("compile Model")

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
        return model




    def compile_model(self,model ,optimizer):
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
        return model


    def fit_model(self,data, model,epoch_batch_size):
        print("fit model")
        predictors = data.drop(["target"],axis=1).to_numpy()#.as_matrix()
        #targets = to_categorical(data.target)

        predictor_scaler= StandardScaler().fit(predictors)

        predictors= predictor_scaler.transform(predictors)

        print(data.target.values)
        print(predictors[1])

        early_stopping_monitor= EarlyStopping(patience=14,monitor="val_accuracy")

        mt =model.fit(predictors,data.target.to_numpy(), epochs=100,batch_size=epoch_batch_size ,
                      validation_split = 0.25,callbacks=[early_stopping_monitor])#

        mlist = [mt]
        #vs.validation_plot(self=vs,model_list=mlist)
        #vs.accuracy_plot(self=vs, model_list=mlist)
        #self.save_model(model)
        return mt,model


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



    def make_prediction_with(self,model_name,data_to_predict_with):
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


    def retrain_model(self,model_name,data):
        loaded_model = load_model(model_name)

        predictors = data.drop(["target"], axis=1).values  # .as_matrix()
        targets = to_categorical(data.target)

        print(data.target.values)
        print(predictors)

        mt = self.model.fit(predictors, data.target.values, epochs=1000,
                            validation_split=0.10)  # , use_multiprocessing=True) This seems to be only for training large pools of models

        # early_stopping_monitor= EarlyStopping(patience=2)
        # self.model.fit(predictors,target,validation_split = 0.3,nb_epoch=20
        # ,callbacks=[early_stopping_monitor])
        mlist = [mt]
        vs.validation_plot(self=vs, model_list=mlist)
        vs.accuracy_plot(self=vs, model_list=mlist)
        self.save_model()
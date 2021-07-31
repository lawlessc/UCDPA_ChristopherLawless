import keras
import pandas as pd
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU ,PReLU,ELU
from keras.utils.np_utils import to_categorical
from keras.models import load_model ,Sequential
from keras.optimizers import SGD ,Adam

from keras.callbacks import EarlyStopping
from sklearn import preprocessing

from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import Conv2D


import matplotlib.pyplot as plt
from datetime import datetime
import visualisers as vs

class NN_definer:

    model = Sequential()

    learning_rate = [.000001,0.01,1]

    def load_model(self, model_name):
        self.model = load_model(model_name)

    def specify_model(self,data):
        print("Specify Model")
        my_optimizer = SGD(learning_rate=0.0000001,momentum=0.3, nesterov=True)
        #my_optimizer = Adam(learning_rate=0.0000333)

        #self.model.add(Dense(3,activation="relu", input_shape=(data.shape[1] - 1,)))
        self.model.add(LeakyReLU(3,input_shape=(data.shape[1] - 1,)))
        #self.model.add(ELU(12289,input_shape=(data.shape[1]-1,)))
       # self.model.add(Dense(12289, activation="relu",input_shape=(data.shape[1]-1,)))
       # self.model.add(LeakyReLU(4096))
        self.model.add(LeakyReLU(3))
        #self.model.add(Dense(3, activation="sigmoid"))
    #    self.model.add(Dense(3, activation="relu"))
       # self.model.add(Dense(42, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))





        print("compile Model")
        self.model.compile(optimizer=my_optimizer, loss='binary_crossentropy', metrics= ["accuracy"])
        # self.model.compile(optimizer='adam', loss='mean_squared_error',metrics=["accuracy"])
        # self.model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
        self.fit_model(data)


    def fit_model(self,data):
        print("fit model")
        predictors = data.drop(["target"],axis=1).values#.as_matrix()
        targets = to_categorical(data.target)

        print(data.target.values)
        print(predictors)

        mt =self.model.fit(predictors,data.target.values, epochs=5000,validation_split = 0.10)#, use_multiprocessing=True) This seems to be only for training large pools of models

        #early_stopping_monitor= EarlyStopping(patience=2)
        #self.model.fit(predictors,target,validation_split = 0.3,nb_epoch=20
        # ,callbacks=[early_stopping_monitor])
        mlist = [mt]
        vs.validation_plot(self=vs,model_list=mlist)
        vs.accuracy_plot(self=vs, model_list=mlist)
        self.save_model()



    def save_model(self):
        current = datetime.now()
        date = current.strftime("%d_%m_%Y_%H_%M_%S")
        self.model.save(date+".h5")


    def trainer_loop(self, predictors, target):
       for lr in self.learning_rate:
           #self.model = get_new_model()
           #optimizer = SGD(lr=lr)
           #self.model.compile(optimizer=optimizer, loss = "categorocal_crossentropy")
           self.model.fit(predictors,target)

    def verify_model_info(self):
        print("Loss: " + self.model.loss)
        self.model.summary()



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
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.models import load_model
#from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn import preprocessing



import matplotlib.pyplot as plt
from datetime import datetime
import visualisers as vs

class NN_definer:

    model = Sequential()

    learning_rate = [.000001,0.01,1]

    def load_model(self, model_name):
        self.model = load_model(model_name)

    def specify_model(self):
        print("Specify Model")
        #predictor columns, only one is need , it outputs 1 or 0
       # n_cols = predictors                     #The input shape is set to the number of datapoints persample
        self.model.add(Dense(11059,activation="relu",input_shape=(12288,)))
        self.model.add(Dense(5529,activation="relu"))
        self.model.add(Dense(1843, activation="relu"))
        #self.model.add(Dense(10, activation="relu"))
        #self.model.add(Dense(1,activation="softmax"))
        self.model.add(Dense(1, activation="sigmoid"))
        print("compile Model")
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics= ["accuracy"])
        # self.model.compile(optimizer='adam', loss='mean_squared_error',metrics=["accuracy"])
        # self.model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])







    def fit_model(self,data):
        print("fit model")
        self.verify_model_info()
        predictors = data.drop(["target"],axis=1).values#.as_matrix()
      #  targets = to_categorical(data.target)

        print(data.target.values)
        #print(predictors)

        mt =self.model.fit(predictors,data.target.values, epochs=40,validation_split = 0.36, use_multiprocessing=True)

        #early_stopping_monitor= EarlyStopping(patience=2)
        #self.model.fit(predictors,target,validation_split = 0.3,nb_epoch=20
        # ,callbacks=[early_stopping_monitor])
        mlist = [mt]
       # self.validation_plot(mlist)
        vs.validation_plot(self=vs,model_list=mlist)



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




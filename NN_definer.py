import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.models import load_model
#from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from datetime import datetime

class NN_definer:

    model = Sequential()

    learning_rate = [.000001,0.01,1]

    def load_model(self, model_name):
        self.model = load_model(model_name)

    def specify_model(self):
        print("Specify Model")
        #predictor columns, only one is need , it outputs 1 or 0
       # n_cols = predictors                     #The inpust shape is set to the number of datapoints persample
        self.model.add(Dense(12288,activation="relu",input_shape=(12288,)))

        self.model.add(Dense(32,activation="relu"))

        self.model.add(Dense(1,activation="softmax"))
        print("compile Model")
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        # self.model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=["accuracy"])
        # self.model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])



    def verify_model_info(self):
        print("Loss: " + self.model.loss)
       # print("Optimizer: " + self.optimizer)



    def fit_model(self,data):
        print("fit model")
        self.verify_model_info()
        predictors = data.drop(["target"],axis=1)#.as_matrix()
        target = to_categorical(data.target)

        #print(predictors)

        self.model.fit(predictors,target,validation_split= 0.3,nb_epoch=20)

        early_stopping_monitor= EarlyStopping(patience=2)
        #self.model.fit(predictors,target,validation_split = 0.3,nb_epoch=20
        # ,callbacks=[early_stopping_monitor])



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
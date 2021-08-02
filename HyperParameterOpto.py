from keras.optimizers import SGD ,Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


import NN_definer as nnd

class hyper_paramenter_training:


    best_model = None
    best_accuracy = None
    neuralnet_d = nnd.NN_definer()

    def train_network(self,data):




        my_optimizer =  SGD(learning_rate=0.0001,momentum=0.99, nesterov=True)
        my_optimizer2 = SGD(learning_rate=0.0000001,momentum=0.99, nesterov=True)
        my_optimizer3 = SGD(learning_rate=0.000001, momentum=0.30, nesterov=False)
        my_optimizer4 = SGD(learning_rate=0.000001, momentum=0.30, nesterov=True)
        my_optimizer5 = SGD(learning_rate=0.00001, momentum=0.30, nesterov=False)
        my_optimizer6 = SGD(learning_rate=0.00001, momentum=0.30, nesterov=True)
        my_optimizer7 = Adam(learning_rate=0.00333)
        my_optimizer8 = Adam(learning_rate=0.0000000333)



        batch_sizes  = [10,100]
        hidden_layers = [1,2]
        hidden_layer_widths = [16,32]
        optimizer_list = [my_optimizer]#,my_optimizer2,"adam",my_optimizer3,"sgd",my_optimizer4,my_optimizer5,my_optimizer6,my_optimizer7,my_optimizer8]




        for index,optimizer in enumerate(optimizer_list):
            for batch_size in batch_sizes:
                for layers in hidden_layers:
                    for widths in hidden_layer_widths:
                        print("optimizer:" +str(index))
                        print("width:"+str(widths))
                        print("layers:" + str(layers+1))#added a number to this to include
                        print("batch size:" + str(batch_size))
                        model = self.neuralnet_d.define_model(data, layers, widths)
                        model = self.neuralnet_d.compile_model(model, optimizer)
                        history , model = self.neuralnet_d.fit_model(data, model, batch_size)

                        self.set_best_model(history,model)





    def set_best_model(self,history,model):
        print(history.history.keys())
        #print("test"+str(history.history["val_accuracy"]))
        self.best_model=model

        if(self.best_model == None):
            self.best_model=model

        if(self.best_accuracy == None):
            self.best_accuracy =history.history["val_accuracy"]

        if(self.best_accuracy < history.history["val_accuracy"]):
            self.best_model = model
            self.best_accuracy = history.history["val_accuracy"]
            print("New Best accuracy"+str(history.history["val_accuracy"]))
            self.neuralnet_d.save_model(model)



from keras.optimizers import SGD ,Adam, RMSprop
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical


import NN_definer as nnd

class GridSearcher:


    neuralnet_d = nnd.NN_definer()


    def do_search(self,data):


        predictors = data.drop(["target"], axis=1).to_numpy()  # .as_matrix() used in datacamp but is deprecated
        #predictor_scaler = StandardScaler().fit(predictors)
        #predictors = predictor_scaler.transform(predictors)

        #predictors.reshape(2,3,2)

        #print(predictors.shape())
        #print(data.target.values)

        #print("test2222222222222  " +str(predictors[0]))


       # my_optimizer = SGD(learning_rate=0.000001, momentum=0.99, nesterov=False)
       # my_optimizer2 = SGD(learning_rate=0.000001, momentum=0.33, nesterov=True)
        # my_optimizer2 = SGD(learning_rate=0.0000001, momentum=0.99, nesterov=True)
        # my_optimizer3 = SGD(learning_rate=0.000001, momentum=0.30, nesterov=False)
        # my_optimizer4 = SGD(learning_rate=0.000001, momentum=0.30, nesterov=True)
        # my_optimizer5 = SGD(learning_rate=0.00001, momentum=0.30, nesterov=False)
        # my_optimizer6 = SGD(learning_rate=0.00001, momentum=0.30, nesterov=True)
        # my_optimizer7 = Adam(learning_rate=0.00333)
        # my_optimizer8 = Adam(learning_rate=0.0000000333)

        #adam1 = Adam(learning_rate=0.333)
        #adam2 = Adam(learning_rate=0.0333)
        #adam3 = Adam(learning_rate=0.00000333)

        #rms1 = RMSprop(learning_rate=0.1)



        model = KerasClassifier(build_fn=self.neuralnet_d.create_model)





        param_grid = {"epochs":[10,25],
                      "first_layer" :[10,100,500,4096],
                      "hidden_layers": [0,1,2,3],
                      "layer_widths": [2,3,4,5],
                       "optimizer": ["adam"],
                       "winit": ["normal"],
                      "batch_size":[10],
                      "dropout":[0.3],
                      "decay":[0.01]
                      }


        grid = GridSearchCV(estimator=model,param_grid= param_grid, n_jobs= 1, cv=2,verbose=3)

        #targets = to_categorical(data.target.values)

        results = grid.fit(predictors,data.target.values)

        print("Best Score: %f " ,results.best_score_)

        print("Best Parameters: %s" , results.best_params_)
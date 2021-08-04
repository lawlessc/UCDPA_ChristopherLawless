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
        predictors = data.drop(["target"], axis=1).to_numpy()
        #predictor_scaler = StandardScaler().fit(predictors)
        #predictors = predictor_scaler.transform(predictors)
        #predictors.reshape(2,3,2)

        model = KerasClassifier(build_fn=self.neuralnet_d.create_model)

        param_grid = {"epochs":[100],
                      "first_layer" :[2],
                      "hidden_layers": [0,1,2,3],
                      "layer_widths": [3,4],
                      "optimizer": ["adam"],
                      "winit": ["normal"],
                      "batch_size":[10,100],
                      "dropout":[0,0.1,0.3],
                      "decay":[0.01]
                      }

        grid = GridSearchCV(estimator=model,param_grid= param_grid, n_jobs= 1, cv=2,verbose=3)

        targets = to_categorical(data.target.values)
        results = grid.fit(predictors,targets)

        print("Best Score: %f " ,results.best_score_)
        print("Best Parameters: %s" , results.best_params_)
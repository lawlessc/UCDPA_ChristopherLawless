from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
from tensorflow.keras import mixed_precision
import numpy
import pandas as pd
import numpy as np




def do_pca_of_data(data):


    try:
        data = data.drop(["target", "id"], axis=1)

    except:
        print("")

    predictors = data.to_numpy()

    # predictor_scaler = MinMaxScaler().fit(predictors)
    # predictors = predictor_scaler.transform(predictors)

    predictor_scaler = StandardScaler().fit(predictors)
    #StandardScaler.set_params()#(with_mean=False)
    predictors = predictor_scaler.transform(predictors)

    pca = PCA(0.95)
    pca.fit(predictors)
    print("PCA")
    print(pca.n_components_)
    return pca.transform(predictors)



import numpy as np
from keras.optimizers import SGD, Adam, Nadam
import visualisers as vs
import NeuralNetDefiner as Nnd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os



class HyperParameterOpto:
    """This is a class for doing hyper parameter optimization"""

    best_model = None
    best_accuracy = None
    neuralnet_d = Nnd.NeuralNetDefiner()
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    def train_network(self, data):
        """This is function takes in pandas dataframe and performs hyper parameter optimization, everytime a model
        beats the previous models on improvement it saves that model """

        # # trying out optimizers with different set learning rates.
        # lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.5, decay_steps=0)
        # opto_ssg3 = SGD(learning_rate=lr_decayed_fn)
        # adam2 = Adam(learning_rate=lr_decayed_fn)
        # tf.device("cpu:0")

        # opt = SGD(lr=0.01)

        batch_sizes = [400]
        seed_list = [69]
        hidden_layers = [1]
        hidden_layer_widths = [16]
        # optimizer_list = ["nadam"]  # ,"adam",my_optimizer3,"sgd",my_optimizer4,my_optimizer5,
        # my_optimizer6,
        # my_optimizer7,my_optimizer8]

        for index, seeds in enumerate(seed_list):
            for batch_size in batch_sizes:
                for layers in hidden_layers:
                    for widths in hidden_layer_widths:
                        print("seed: "+ str(seeds))
                        print("optimizer:" + "nadam")
                        print("width:" + str(widths))
                        print("layers:" + str(layers + 1))  # added a number to this to include
                        print("batch size:" + str(batch_size))

                        model = self.neuralnet_d.create_model(optimizer="nadam",seed_num=seeds)

                        history, model = self.neuralnet_d.fit_model(data=data, epochs=122, model=model,
                                                                    batch_size=batch_size, use_early_stopping_time=0)

                        mlist = [history]
                        vs.validation_plot(mlist)
                        vs.accuracy_plot(mlist)



                    self.set_best_model(history, model,data)



    def train_loaded_network(self, data,model):
        """This is function takes in pandas dataframe and performs hyper parameter optimization, everytime a model
        beats the previous models on improvement it saves that model """

        # # trying out optimizers with different set learning rates.
        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.05, decay_steps=25)
        # opto_ssg3 = SGD(learning_rate=lr_decayed_fn)
        adam2 = Adam(learning_rate=lr_decayed_fn)
        az= Nadam(learning_rate=0.54)
        # tf.device("cpu:0")

        # opt = SGD(lr=0.01)

        batch_sizes = [256]
        seed_list = [69]
        hidden_layers = [1]
        hidden_layer_widths = [16]
        # optimizer_list = ["nadam"]  # ,"adam",my_optimizer3,"sgd",my_optimizer4,my_optimizer5,
        # my_optimizer6,
        # my_optimizer7,my_optimizer8]

        for index, seeds in enumerate(seed_list):
            for batch_size in batch_sizes:
                for layers in hidden_layers:
                    for widths in hidden_layer_widths:
                        print("seed: "+ str(seeds))
                        print("optimizer:" + "nadam")
                        print("width:" + str(widths))
                        print("layers:" + str(layers + 1))  # added a number to this to include
                        print("batch size:" + str(batch_size))

                        # model = self.neuralnet_d.create_model(optimizer=adam2,seed_num=seeds)
                        model.compile(optimizer=az, loss='binary_crossentropy', metrics=["accuracy"])

                        history, model = self.neuralnet_d.fit_model(data=data, epochs=120, model=model,
                                                                    batch_size=batch_size, use_early_stopping_time=0)

                        mlist = [history]
                        vs.validation_plot(mlist)
                        vs.accuracy_plot(mlist)



                    self.set_best_model(history, model,data)

    def train_auto_encoder_network(self, data):
        """This is function takes in pandas dataframe and performs hyper parameter optimization, everytime a model
        beats the previous models on improvement it saves that model """

        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.99999, decay_steps=200)
        opto_ssg3 = SGD(learning_rate=lr_decayed_fn)

        batch_sizes = [600]
        hidden_layers = [2]
        hidden_layer_widths = [50]
        adam2 = Adam(learning_rate=lr_decayed_fn)
        optimizer_list = [opto_ssg3, adam2]

        # my_optimizer6,
        # my_optimizer7,my_optimizer8]

        for index, optimizer in enumerate(optimizer_list):
            for batch_size in batch_sizes:
                for layers in hidden_layers:
                    for widths in hidden_layer_widths:
                        print("optimizer:" + str(index))
                        print("width:" + str(widths))
                        print("layers:" + str(layers))  # added a number to this to include
                        print("batch size:" + str(batch_size))

                        model = self.neuralnet_d.create_autoencoder(self, hidden_layers=1, max_pool_size=10,
                                                                    layer_widths=10,
                                                                    optimizer=optimizer,
                                                                    lossf= "mean_squared_error",
                                                                    winit="gl_uniform", cnn_window_size=30,
                                                                    dropout=0.01, decay=0.01,
                                                                    input_shape=(12288,), )

                        history, model = self.neuralnet_d.fit_model_auto_encoder(data=data, epochs=40, model=model,
                                                                                 batch_size=batch_size,
                                                                                 use_early_stopping_time=0)

                        mlist = [history]
                        vs.validation_plot(mlist)
                        vs.accuracy_plot(mlist)
                        self.set_best_model(history, model,data)

    def set_best_model(self, history, model,data):
        """This class evaluates a model against the last best model, if the new model is better it is saved and takes
        the previous models places as the best model """
        print(history.history.keys())

        predictors = data.drop(["id"], axis=1)
        predictors = predictors.drop(["target"], axis=1).to_numpy()
        data = data.to_numpy()
        self.neuralnet_d.save_model(model)

        num_rows, num_cols = predictors.shape
        predictors = np.reshape(predictors, (num_rows, 4096, 3))


        ##predictors = data

        # predictor_scaler = MinMaxScaler(feature_range=(0, 1)).fit(predictors)
        # predictors = predictor_scaler.transform(predictors)

        ynew = model.predict(predictors)

        print(ynew)

        model.summary()

        print("ynew:"+ str(len(ynew)))
        # show the inputs and predicted outputs
        for i in range(10):  # only test the first 10 or so
            print("X=%s, Predicted=%s" % (data[i], ynew[i]))


        # print("test"+str(history.history["val_accuracy"]))
        #self.best_model = model

        if self.best_model is None:
            self.best_model = model
            self.best_accuracy = history.history["val_accuracy"]


        # This if statement is discrete from the others the statement below it will attempt to evaluate a NoneType and
        # crash
        if self.best_accuracy is None:
            self.best_accuracy = history.history["val_accuracy"]

        if self.best_accuracy < history.history["val_accuracy"]:
            self.best_model = model
            self.best_accuracy = history.history["val_accuracy"]
            print("New Best accuracy" + str(history.history["val_accuracy"]))
            self.neuralnet_d.save_model(model)

            #print("Predict")


    def make_predictions(self, model,data):
        predictors = data.drop(["id"], axis=1)
        predictors = predictors.drop(["target"], axis=1).to_numpy()

        num_rows, num_cols = predictors.shape
        predictors = np.reshape(predictors, (num_rows, 4096, 3))
        ynew = model.predict(predictors)

        print(ynew)






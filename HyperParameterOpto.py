from keras.optimizers import SGD, Adam
import visualisers as vs
import NeuralNetDefiner as Nnd


class HyperParameterOpto:
    """This is a class for doing hyper parameter optimization"""

    best_model = None
    best_accuracy = None
    neuralnet_d = Nnd.NeuralNetDefiner()

    def train_network(self, data):
        """This is function takes in pandas dataframe and performs hyper parameter optimization, everytime a model
        beats the previous models on improvement it saves that model """

        #trying out optimizers with different set learning rates.
        my_optimizer = SGD(learning_rate=0.0001, momentum=0.99, nesterov=True)
        my_optimizer2 = SGD(learning_rate=0.0000001, momentum=0.99, nesterov=True)
        # my_optimizer3 = SGD(learning_rate=0.000001, momentum=0.30, nesterov=False)
        # my_optimizer4 = SGD(learning_rate=0.000001, momentum=0.30, nesterov=True)
        # my_optimizer5 = SGD(learning_rate=0.00001, momentum=0.30, nesterov=False)
        # my_optimizer6 = SGD(learning_rate=0.00001, momentum=0.30, nesterov=True)
        # my_optimizer7 = Adam(learning_rate=0.00333)
        # my_optimizer8 = Adam(learning_rate=0.0000000333)

        batch_sizes = [10, 100]
        hidden_layers = [1, 2]
        hidden_layer_widths = [16, 32]
        optimizer_list = [my_optimizer, my_optimizer2]  # ,"adam",my_optimizer3,"sgd",my_optimizer4,my_optimizer5,
        # my_optimizer6,
        # my_optimizer7,my_optimizer8]

        for index, optimizer in enumerate(optimizer_list):
            for batch_size in batch_sizes:
                for layers in hidden_layers:
                    for widths in hidden_layer_widths:
                        print("optimizer:" + str(index))
                        print("width:" + str(widths))
                        print("layers:" + str(layers + 1))  # added a number to this to include
                        print("batch size:" + str(batch_size))
                        model = self.neuralnet_d.define_model(data, layers, widths)
                        model = self.neuralnet_d.compile_model(model, optimizer)
                        history, model = self.neuralnet_d.fit_model(data=data, epochs=36, model=model,
                                                                    batch_size=batch_size, use_early_stopping_time=3)

                        mlist = [history]
                        vs.validation_plot(mlist)
                        vs.accuracy_plot(mlist)
                        self.set_best_model(history, model)

    def set_best_model(self, history, model):
        """This class evaluates a model against the last best model, if the new model is better it is saved and takes
        the previous models places as the best model """
        print(history.history.keys())
        # print("test"+str(history.history["val_accuracy"]))
        self.best_model = model

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

from keras.optimizers import SGD ,Adam

import NN_definer as nnd


def train_network(data_list):
    #learning_rate = [.000001, 0.01, 1]
    neuralnet_d = nnd.NN_definer()
    model =neuralnet_d.define_model(data_list[0])

    my_optimizer = SGD(learning_rate=0.001,momentum=0.99, nesterov=True)
    my_optimizer2 = SGD(learning_rate=0.000001, momentum=0.99, nesterov=True)
    my_optimizer3 = SGD(learning_rate=0.001, momentum=0.99, nesterov=True)
    my_optimizer4 = SGD(learning_rate=0.01, momentum=0.56, nesterov=True)
    my_optimizer5 = SGD(learning_rate=0.00000001, momentum=0.33, nesterov=False)
    my_optimizer6 = Adam(learning_rate=0.00000333)
    my_optimizer7 = Adam(learning_rate=0.00333)
    my_optimizer8 = Adam(learning_rate=0.0000000333)



    optimizer_list = [my_optimizer,my_optimizer2,"adam",my_optimizer3,"sgd",my_optimizer4,my_optimizer5,my_optimizer6,my_optimizer7,my_optimizer8]

    for data in data_list:
        for optimizer in optimizer_list:
            model = neuralnet_d.compile_model(model, optimizer)
            neuralnet_d.fit_model(data, model)

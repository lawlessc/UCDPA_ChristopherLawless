import matplotlib.pyplot as plt


def validation_plot(self ,model_list):


    for model in model_list:
        plt.plot(model.history["val_loss"] ,'r')


    plt.xlabel('Epochs')
    plt.ylabel('Validation score')
    plt.show()

def accuracy_plot(self ,model_list):


    for model in model_list:
        plt.plot(model.history["val_accuracy"] ,'b')


    plt.xlabel('Epochs')
    plt.ylabel('Accuracy score')
    plt.show()
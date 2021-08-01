import matplotlib.pyplot as plt


def validation_plot(self ,model_list):


    for model in model_list:
        plt.plot(model.history["loss"], 'r' ,  label="loss")
        plt.plot(model.history["val_loss"] ,'b', label="validation loss")



    plt.xlabel('Epochs')
    plt.ylabel('Validation score')
    plt.legend()
    plt.show()

def accuracy_plot(self ,model_list):


    for model in model_list:
        plt.plot(model.history["accuracy"], 'r', label="accuracy")
        plt.plot(model.history["val_accuracy"] ,'b',  label="validation accuracy")


    plt.xlabel('Epochs')
    plt.ylabel('Accuracy score')
    plt.legend()
    plt.show()
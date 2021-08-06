import matplotlib.pyplot as plt
import numpy as np

from scipy import signal
from scipy.fft import fftshift

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


def signal_plotter(self, signals_list):

    for signals in signals_list:
        plt.plot(signals, label="signal")

    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()

     #The blackhole colission should be represented by a high rising frequency so maybe FFT can remove it.
def fftPlot(self, xf,yf):


    plt.plot(xf, np.abs(yf))

    plt.xlabel('Frequency')
    #plt.ylabel('Frequency')
    #plt.legend()
    plt.show()

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
def spectralplot(self,data):


    f, t, Sxx = signal.spectrogram(data,4096)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.show()




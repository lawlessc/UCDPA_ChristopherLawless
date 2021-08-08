import matplotlib.pyplot as plt
import numpy as np

from scipy import signal
from scipy.fft import fftshift
from scipy.fft import rfft, rfftfreq , irfft

def validation_plot(self ,model_list):
    '''This takes in the model history output, and plots the loss and validation testing loss of the model'''

    for model in model_list:
        plt.plot(model.history["loss"], 'r' ,  label="loss")
        plt.plot(model.history["val_loss"] ,'b', label="validation loss")



    plt.xlabel('Epochs')
    plt.ylabel('Validation score')
    plt.legend()
    plt.show()

def accuracy_plot(self ,model_list):
    '''This takes in the model history output, and plots the accuracy and validation testing accuracy of the model'''

    for model in model_list:
        plt.plot(model.history["accuracy"], 'r', label="accuracy")
        plt.plot(model.history["val_accuracy"] ,'b',  label="validation accuracy")


    plt.xlabel('Epochs')
    plt.ylabel('Accuracy score')
    plt.legend()
    plt.show()


def signal_plotter(self, signals_list):
    '''This takes a list of ndarrays containing signals for ligo and displays them horizontally'''

    for signals in signals_list:
        plt.plot(signals, label="signal")

    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()

#The blackhole colission should be represented by a high rising frequency so maybe FFT can remove it.
#I learnt to use this here https://realpython.com/python-scipy-fft/
def fftPlot(self, data):
    '''This plots a fast fourrier transform of the sample'''
    y = rfft(data)
    x = rfftfreq(4096, 1 / 2048)
    plt.xlabel("Frequency")
    #plt.plot(x,y)

    points_per_freq = len(x) / (2048 / 2)
    for frequency in range(600,1500):
         target = int(points_per_freq * frequency)
         y[target - 1: target + 2] = 0
    y = irfft(y)

    # for frequency in range(3500,4000):
    #      target = int(points_per_freq * frequency)
    #      y[target - 1: target + 2] = 0
    # y = rfft(y)
    plt.plot(y)




    plt.show()


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
def spectralplot(self,data):
    '''This shows a spectral plot of the data input into it.'''
    f, t, Sxx = signal.spectrogram(data,4096)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.show()




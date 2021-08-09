import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import fftshift
from scipy.fft import rfft, rfftfreq, irfft


def validation_plot(model_hist):
    """This takes in the model history output, and plots the loss and validation testing loss of the model"""
    for model in model_hist:
        plt.plot(model.history["loss"], 'r', label="loss")
        plt.plot(model.history["val_loss"], 'b', label="validation loss")

    plt.xlabel('Epochs')
    plt.ylabel('Validation score')
    plt.legend()
    plt.show()


def accuracy_plot(model_hist):
    """This takes in the model history output, and plots the accuracy and validation testing accuracy of the model"""
    for model in model_hist:
        plt.plot(model.history["accuracy"], 'r', label="accuracy")
        plt.plot(model.history["val_accuracy"], 'b', label="validation accuracy")

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy score')
    plt.legend()
    plt.show()


def signal_plotter(signals_list):
    """This takes a list of ndarrays containing signals for ligo and displays them horizontally"""

    for signals in signals_list:
        plt.plot(signals, label="signal")

    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()


# The blackhole colission should be represented by a high rising frequency so maybe FFT can remove it.
# I learnt to use this here https://realpython.com/python-scipy-fft/
def fft_plot(data):
    """This plots a fast fourier transform of a sample"""
    y = rfft(data)
    x = rfftfreq(4096, 1 / 2048)  # 2048 is the sampling rate
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.plot(x, y)
    plt.show()


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
def spectral_plot(data):
    """This shows a spectral plot of the data input into it."""
    f, t, Sxx = signal.spectrogram(data, 2096)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def spectral_plot_multiple(data):
    '''This shows a spectral plot of the data input into it.'''
    for signals in data:
        f, t, Sxx = signal.spectrogram(signals, 2096)
        plt.pcolormesh(t, f, Sxx, shading='gouraud', alpha=0.6)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def targets_bar_chart(data):
    '''This shows a bar chart of the data input into it.'''
    counts = data['target'].value_counts()
    plt.ylabel("number of values")
    plt.bar(["0", "1"], height=counts, width=0.5, color=["b", "g"])
    plt.show()

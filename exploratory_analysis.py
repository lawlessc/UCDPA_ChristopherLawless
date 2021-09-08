import pandas as pd
import matplotlib.pyplot as plt
import import_ligo_samples as lg
import import_targets as tg
import visualisers as vs


def analysis_of_data():
    """ This imports samples 10 to 100 as a dataframe and displays some details on them as via print"""
    df = lg.import_flat_samples_add_targets(0, 10)
    print(df.shape)
    print(df.columns)
    print(df.head)


def analysis_of_signal():
    """ This imports samples 10 to 100 as a dataframe and displays some details on them as via print"""
    sample = lg.import_single_sample_as_ndarray("00000e74ad")
    signal_list = [sample[0, :], sample[1, :], sample[2, :]]
    vs.signal_plotter(signals_list=signal_list)
    vs.fft_plot(sample[0, :])
    vs.spectral_plot(data=sample[0, :])

    df = lg.import_flat_samples_add_targets(0, 10)
    df.shape
    df.columns
    df.head


def analysis_of_targets():
    """ This imports all the target values and ID as a dataframe, prints information on them and displays a histogram
    of their values """
    df = tg.import_training_targets()
    df.shape
    df.columns
    df.head
    vs.targets_bar_chart(df)

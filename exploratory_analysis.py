import pandas as pd
import matplotlib.pyplot as plt
import import_ligo_samples as lg
import import_targets as tg

def analysis_of_data():
    ''' This imports samples 10 to 100 as a dataframe and displays some details on them as via print'''
    df = lg.import_many_flat_samples_add_targets(10,100)
    print(df.shape)
    print(df.columns)
    print(df.head)

def analysis_of_targets():
    ''' This imports all the target values and ID as a dataframe, prints information on them and displays a histogram of their values'''
    df =tg.import_training_targets()
    print(df.shape)
    print(df.columns)
    print(df.head)
    counts = df['target'].value_counts()
    plt.ylabel("number of values")
    plt.hist(counts, bins=3, histtype='bar')#
    plt.show()


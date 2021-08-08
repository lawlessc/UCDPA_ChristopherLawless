#import re

# This just uses the filenames of samples to get the path to the data
# This file works on the assumption the user has the relevant Ligo data
# placed in the "data" in the same form it was downloaded via the kaggle API
# initially i tried to do this with regex, it was easier not to use regex
def get_path(ligo_filename):
    first_three= ligo_filename[0:3]
    print(first_three)
    file_path = first_three[0]+"/"+first_three[1]+"/"+first_three[2]+"/"+ligo_filename+".npy"
    print(file_path)
    return file_path

def get_path_training(value):
    file_path = "data/train/"+ get_path(value)
    print(file_path)
    return file_path

def get_path_testing(value):
    file_path = "data/test/" + get_path(value)
    return file_path


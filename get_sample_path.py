# This just uses the filenames of samples to get the path to the data
# This file works on the assumption the user has the relevant LIGO data
# placed in the "data" in the same form it was downloaded via the kaggle API
# initially i tried to do this with regex, it was easier and faster not to use regex
def get_path(id):
    """This takes a single string of the ID and outputs the path """
    first_three = id[0:3]
    # print(first_three)
    file_path = first_three[0] + "/" + first_three[1] + "/" + first_three[2] + "/" + id + ".npy"
    # print(file_path)
    return file_path


def get_path_training(id):
    """This takes a single string of the id, get's the path and adds the training folder path to the path """
    file_path = "data/train/" + get_path(id)
    # print(file_path)
    return file_path


def get_path_testing(value):
    """This takes a single string of the id, get's the path and adds the test folder path to the path """
    file_path = "data/test/" + get_path(value)
    return file_path

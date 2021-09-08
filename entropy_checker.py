from scipy.stats import entropy




def try_entropy(data):
    print("Start Entropy")
    e = entropy(data[0])
    print("Entropy")
    print(e)
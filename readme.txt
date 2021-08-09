This project was created using python 3.7

This project attempts to use neural networks via the Keras API to classify if LIGO signals contain gravitational waves
from the collisions of blackholes.



Some recommendations:
The Ligo data is >70gbs and not included. When you download the data
i recommend you turn OFF indexing in pycharm for the "data" folder as
indexing it takes well over and hour and doesn't serve any useful purpose with these files.

The same applies for git/github, don't include or add it.

Loading the data is slow and uses a lot of memory. I was doing this with a Tensorflow enabled GPU

I have included the first 154 entries if a user wants to run it immediatly


Necessary imports etc:
Sklearn (fft,scaling)
Scipy.signal (Spectrograms)
Keras (Neural Networks)
TensorFlow (Backend for building and training Neural Networks)
Matplotlib (visualizers)
Matplotlib.pyplot  (visualizers)
Kaggle  (Retrieving Competition Data)

This project is also using Python Version 3.7 and CUDA

The G2net dataset https://www.kaggle.com/c/g2net-gravitational-wave-detection/data


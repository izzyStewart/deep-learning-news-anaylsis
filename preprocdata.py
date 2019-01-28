# Import general modules
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from collections import Counter
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.datasets import make_classification

# Import Keras modules 
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras import models
from keras.models import Model
from keras import Input
from keras import layers
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasClassifier


def read_csv(file, col1, col2, col3):
    """Function to read in csv files and turn them into arrays.
    """
    df = pd.read_csv(file, usecols=[col1,col2, col3])
    return df


def text_to_num(df):
    """Function turns words into assigned number. 
    """
    # df to array
    array = df['relevance'].values
    # Takes an array of text strings and turns them into lists of words as individual items
    for i in range(len(array)):
        array[i] = text_to_word_sequence(array[i])
    labels = sum(array, [])
    labels = [w.replace('yes', '1') for w in labels]
    labels = [w.replace('no', '0') for w in labels]
    labels = np.array(labels)
    labels = labels.astype(int)
    return labels

def create_input_vectors(max_words, input_val, max_len):
    array = input_val.values
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(array)
    sequences = tokenizer.texts_to_sequences(array)
    words = tokenizer.word_index
    input_vector = pad_sequences(sequences, maxlen= max_len)
    return input_vector, words

def shuffle_data(input1, input2, output):
    indices = np.arange(input1.shape[0])
    np.random.shuffle(indices)
    input1_data = input1[indices]
    input2_data = input2[indices]
    output_data = output[indices]
    return input1_data, input2_data, output_data
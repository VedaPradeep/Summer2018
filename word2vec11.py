import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils
import tensorflow as tf

# From corpus to center/target word & context words

def tokenize(corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus) #list of texts to train
    #["my name is is ","lokesh , well"]
    corpus_tokenized = tokenizer.texts_to_sequences(["my name is Pradeep","lokesh enjoying"])# return sequence for each text
    V = len(tokenizer.word_index)
    #print(type(corpus_tokenized))
    #"""
    print(tokenizer.word_index)
    for words in corpus_tokenized:
        print(words)

    print(V)
    #"""
    return corpus_tokenized,V

corpus = ["my name is Pradeep,his frnd name is lokesh","he is doing well","he is enjoying his life"]
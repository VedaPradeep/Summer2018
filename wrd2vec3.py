import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils
import tensorflow as tf

# From corpus to center/target word & context words

def tokenize(corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus) #list of texts to train
    corpus_tokenized = tokenizer.texts_to_sequences(["my name is is ","lokesh"])# return sequence for each text
    V = len(tokenizer.word_index)
    #print(type(corpus_tokenized))
    #"""
    print(tokenizer.word_index)
    for words in corpus_tokenized:
        print(words)

    print(V)
    #"""
    return corpus_tokenized,V

corpus_tokenized,V = tokenize(["my name is Pradeep,his frnd name is lokesh","he is doing well"])
window_size =2

# Converting class vector integers to binary matrix -- case 1 using pointers
"""
def to_categorical(y,num_classes=None):
    y = np.array(y,dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical =np.zeros(n,num_classes)
    categorical[np.arrange(n),y] =1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical,output_shape)
    return categorical

"""

# Converting class vector integers to binary matrix
def to_categorical_context(x,V) : # x is context words , y is center word
    l = len(x)
    np1 = np.zeros((l*V),dtype='int')
    np1.reshape(l,V)
    for i,num in x :
        np1[i,x[i]] = 1
        return np1


def to_categorical_center(y,V) :
    np2 = np.zeros((V,),dtype='int')
    np2[y] = 1
    return np2

#Converting corpus text to context and center words

def corpus2io(corpus_tokenized,V, window_size):
    for words in corpus_tokenized:
        L = len(words)
        for index,word in enumerate(words):
            #print(index,word)
            context=[]
            start = index - window_size
            end   = index +window_size +1
            #context.append(words[i]-1 for i in range(start,end) if 0<=i<L and i!=index ) -- case1 using pointers
            for i in range(start, end):
                if(0 <= i < L and i != index):
                    context.append(words[i] - 1)
            center = word-1
            #print("pra")
            #print(np.ravel(context))
            #print(np.ravel(center))
            print("test 1")
            x=to_categorical_context(context,V)
            y = to_categorical_center(center, V)
            print("test 2")
            #yield (x,y)
            #yield (x,np.ravel(y))

temp = corpus2io(corpus_tokenized,V, window_size)
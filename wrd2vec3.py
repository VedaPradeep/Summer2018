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
    corpus_tokenized = tokenizer.texts_to_sequences(["my name is Pradeep,his frnd name is lokesh","he is doing well"])# return sequence for each text
    V = len(tokenizer.word_index)
    #print(type(corpus_tokenized))
    #"""
    print(tokenizer.word_index)
    for words in corpus_tokenized:
        print(words)

    print(V)
    #"""
    return corpus_tokenized,V

corpus = ["my name is Pradeep,his frnd name is lokesh","he is doing well"]
corpus_tokenized,V = tokenize(corpus)
window_size =2

# Converting class vector integers to binary matrix -- case 1 using pointers
#"""
def to_categorical(y,num_classes=None):
    y = np.array(y,dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical =np.zeros((n,num_classes))
    categorical[np.arange(n),y] =1
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


def to_categorical(z,V) :
    np2 = np.zeros((V,),dtype='int')
    np2[y] = 1
    return np2
"""

#Converting corpus text to context and center words

def corpus2io(corpus_tokenized,V, window_size):
    for words in corpus_tokenized:
        L = len(words)
        #index=0
        for index,word in enumerate(words):
            #print(index,word)
            context=[]
            labels = []
            start = index - window_size
            end   = index +window_size +1
            #context.append(words[i]-1 for i in range(start,end) if 0<=i<L and i!=index ) -- case1 using pointers
            for i in range(start, end):
                if(0 <= i < L and i != index):
                    context.append(words[i] - 1)
            labels.append(word-1)
            #print(center[0])
            #print("pra")
            #print(np.ravel(context))
            #print(np.ravel(center))
            #print("test 1")
            #In order to convert integer targets into categorical targets, you can use the Keras utility to_categorical
            x=to_categorical(context,V)
            y = to_categorical(labels, V)
            #print("test 2")
            #yield (x,y)
            yield (x,np.ravel(y))

#temp = corpus2io(corpus_tokenized,V, window_size)




# Simple softmax function
def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum(axis=0)


# continuous bagf of words (CBOW) implementation

def cbow(context, label, W1,W2, loss ):
    x = np.mean(context,axis=0)
    h = np.dot(W1.T , x)
    u = np.dot(W2.T , h)
    y_pred = softmax(u)

    e = -label + y_pred
    dW2 = np.outer(h, e)
    dW1 = np.outer(x, np.dot(W2,e))

    W1_new = W1 - eta * dW1  # updating W1
    W2_new = W2 - eta *dW2  # updating W2

    loss += -float(u[label==1]) +np.log(np.sum(np.exp(u))) #loss function

    return W1_new,W2_new,loss


# Initializing values
eta = 0.1 #learning rate
N = 2
loss = 0

np.random.seed(100)
W1 = np.random.rand(V,N)
W2 = np.random.rand(N,V)

for i, (context,label) in enumerate(corpus2io(corpus_tokenized, V, window_size)):
    print(i, "\n center word =", context, "\n context words =\n",label)
    W1,W2,loss = cbow(context,label,W1,W2,loss)
    print("Training example #{} \n-------------------- \n\n \t label = {}, \n \t context = {}".format(i, label, context))
    print("\t W1 = {}\n\t W2 = {} \n\t loss = {}\n".format(W1, W2, loss))

"""
#Training and testing word2vec cbow
import gensim
from gensim.models import Word2Vec
cbow = Word2Vec()
"""

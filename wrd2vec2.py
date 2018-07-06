import keras
import  numpy as np

vocab_size = 1000
data,count,dictionary , reverse_dictionary = collect_data(vocabulary_size = vocab_size)

# Constants and the validation set
window_size = 3  # around the target word that will be used to draw the context words from
vector_dim = 300 #no of nodes in linear activations #our embedding layer will be of size 10,000 x 300
epochs = 1000000  # this designates the number of training iterations we are going to run.

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

model = Model(input= [input])

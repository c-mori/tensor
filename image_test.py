from __future__ import print_function
import numpy as np
import cPickle as pickle
import fcnet as fc


# First reload the data we generated in _notmist.ipynb_.

# In[3]:

pickle_file = 'udacity/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    # test_dataset = save['test_dataset']
    # test_labels = save['test_labels']
    del save  # hint to help gc free up memory

# Reformat into a shape that's more adapted to the models we're going to train:
# - data as a flat matrix,
# - labels as float 1-hot encodings.

# In[4]:

image_size = 28
num_labels = 10


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


x_tr, y_tr = reformat(train_dataset, train_labels)
x_vl, y_vl = reformat(valid_dataset, valid_labels)

print('\nTraining set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
# print('Test set', test_dataset.shape, test_labels.shape)

fc.train(x_tr, y_tr, x_vl, y_vl,
         layer_sizes=[256, 256, 256],
         epochs=8,
         dropouts=[.325],
         learn_rate=1e-3,
         l2_lambda=1e-5,
         batch_size=96,
         classify=True,
         name='image_test')

"""
Where tensorflow model is initialized to the exact point before it is to be trained and tested.

convolutional network
"""
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool1D, MaxPool2D, Dropout, Flatten, Softmax
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np 
import pickle 
from sklearn.preprocessing import MinMaxScaler, normalize
from DeepVCF import Preprocess


physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('no GPU')

with open('/home/tmsincomb/Dropbox/git/DeepVCF/data/test.pickle', 'rb') as infile:
    preprocess = pickle.load(infile)
# with open('/home/tmsincomb/Dropbox/git/DeepVCF/jupyter_nb/hg38.preprocess.pickle', 'rb') as infile:
#     preprocess = pickle.load(infile)
x_train, y_train, pos_array = preprocess.get_training_array(window_size=13)


def loss_func(y_true, y_pred):
    return tf.reduce_sum(tf.pow(y_true - y_pred, 2 )) 


### MODEL ###
inputs = Input(shape=x_train.shape[1:])

x = Conv2D(filters=32, kernel_size=(4,4), input_shape=x_train.shape[1:], activation='elu', padding="same", strides=1)(inputs)
x = MaxPool2D(pool_size=(2, 2), strides=1)(x)
x = Conv2D(filters=64, kernel_size=(3,3), activation='elu', padding="same", strides=1)(x)
x = MaxPool2D(pool_size=(2, 2), strides=1)(x)
x = Conv2D(filters=128, kernel_size=(2,2), activation='elu', padding="same", strides=1)(x)
x = MaxPool2D(pool_size=(2, 2), strides=1)(x)

x = Flatten()(x)

h = Dense(13*16, activation='elu')(x)
dropout = Dropout(0.2)(h)
h = Dense(13*8, activation='elu')(x)
dropout = Dropout(0.2)(h)
h = Dense(13*4, activation='elu')(dropout)
dropout = Dropout(0.2)(h)
h = Dense(13*2, activation='elu')(dropout)
dropout = Dropout(0.2)(h)

Y1 = Dense(4, activation='sigmoid', name='Y1')(dropout)

# x = Conv2D(filters=32, kernel_size=(4,4), input_shape=x_train.shape[1:], activation='elu', padding="same", strides=1)(inputs)
# x = MaxPool2D(pool_size=(2, 2), strides=1)(x)
# x = Conv2D(filters=64, kernel_size=(3,3), activation='elu', padding="same", strides=1)(x)
# x = MaxPool2D(pool_size=(2, 2), strides=1)(x)
# x = Conv2D(filters=128, kernel_size=(2,2), activation='elu', padding="same", strides=1)(x)
# x = MaxPool2D(pool_size=(2, 2), strides=1)(x)

# x = Flatten()(x)

# h = Dense(13*8, activation='elu')(x)
# dropout = Dropout(0.5)(h)
# h = Dense(13*4, activation='elu')(dropout)
# dropout = Dropout(0.2)(h)
# h = Dense(13*2, activation='elu')(dropout)
# dropout = Dropout(0.2)(h)

Y2 = Dense(4, activation='elu', name='Y2')(dropout)
Y4 = Softmax(name='Y4')(Y2)

model = Model(inputs=inputs, outputs=[Y1, Y4])

optim = keras.optimizers.Adam(lr=0.0005)
metrics = ["accuracy"]
losses = {
    "Y1": loss_func,
    'Y4': 'categorical_crossentropy' # 'categorical_crossentropy'
}
model.compile(loss=losses, optimizer=optim, metrics=metrics, loss_weights=[.5, .5])   
##loss_weights=[0.6, 0.4]

model.summary()

y = {
    'Y1':y_train[:,:4],
    'Y4':y_train[:,4:]
}
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
model.fit(x_train, y, validation_split=.1, epochs=20, callbacks=[early_stop])

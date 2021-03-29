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
from DeepVCF import Preprocess

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('no GPU')


def loss_func(y_true, y_pred):
    return tf.reduce_sum(tf.pow(y_true - y_pred, 2 )) 

def ignore_accuracy_of_class(class_to_ignore=0):
    def ignore_acc(y_true, y_pred):
        y_true_class = keras.argmax(y_true, axis=-1)
        y_pred_class = keras.argmax(y_pred, axis=-1)

        ignore_mask = keras.cast(keras.not_equal(y_pred_class, class_to_ignore), 'int32')
        matches = keras.cast(keras.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = keras.sum(matches) / keras.maximum(keras.sum(ignore_mask), 1)
        return accuracy

    return ignore_acc

### MODEL ###
def build_model(x_train, y_train):
    inputs = Input(shape=x_train.shape[1:])

    x = Conv2D(filters=32, kernel_size=(4,4), input_shape=x_train.shape[1:], activation='elu', padding="same", strides=1)(inputs)
    x = MaxPool2D(pool_size=(2, 2), strides=1)(x)
    x = Conv2D(filters=64, kernel_size=(3,3), activation='elu', padding="same", strides=1)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=1)(x)
    x = Conv2D(filters=128, kernel_size=(2,2), activation='elu', padding="same", strides=1)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=1)(x)

    x = Flatten()(x)

    # h = Dense(x_train.shape[1]*16, activation='elu')(x)
    # dropout = Dropout(0.1)(h)
    # h = Dense(x_train.shape[1]*8, activation='elu')(x)
    # dropout = Dropout(0.1)(h)
    # h = Dense(x_train.shape[1]*4, activation='elu')(dropout)
    # dropout = Dropout(0.1)(h)
    # h = Dense(x_train.shape[1]*2, activation='elu')(dropout)
    # dropout = Dropout(0.1)(h)

    h = Dense(13*64, activation='elu')(x)
    dropout = Dropout(0.2)(h)
    h = Dense(13*32, activation='elu')(x)
    dropout = Dropout(0.2)(h)
    h = Dense(13*16, activation='elu')(dropout)
    dropout = Dropout(0.2)(h)
    h = Dense(13*8, activation='elu')(dropout)
    dropout = Dropout(0.2)(h)
    # h = Dense(13*2, activation='elu')(dropout)
    # dropout = Dropout(0.2)(h)

    Y1 = Dense(4, activation='sigmoid', name='Y1')(dropout)
    Y2 = Dense(4, activation='elu', name='Y2')(dropout)
    Y4 = Softmax(name='Y4')(Y2)

    model = Model(inputs=inputs, outputs=[Y1, Y4])

    optim = keras.optimizers.Adam(lr=0.0005)
    metrics = ["accuracy"]
    losses = {
        "Y1": 'binary_crossentropy',
        'Y4': 'categorical_crossentropy'
    }
    model.compile(loss=losses, optimizer=optim, metrics=metrics, loss_weights=[1, {0:1, 1:1, 2:0, 3:1}])   
    model.summary()

    y = {
        'Y1':y_train[:,:4],
        'Y4':y_train[:,4:]
    }
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(x_train, y, validation_split=.3, epochs=100, callbacks=[early_stop])
    
    return model


if __name__ == '__main__':
    # with open('/home/tmsincomb/Dropbox/git/DeepVCF/data/test2.pickle', 'rb') as infile:
    #     preprocess = pickle.load(infile)
    with open('/home/tmsincomb/Dropbox/git/DeepVCF/jupyter_nb/hg38.preprocess.pickle', 'rb') as infile:
        preprocess = pickle.load(infile)
    x_train, y_train, pos_array = preprocess.get_training_array(window_size=15)
    
    # import sys
    # sys.path.append('/home/tmsincomb/Dropbox/thesis/VariantNET/')
    # import variantNet.utils as utils
    # x_train, y_train, pos_array = \
    # utils.get_training_array("/home/tmsincomb/Dropbox/thesis/VariantNET/wd/aln_tensor_chr21", 
    #                          "/home/tmsincomb/Dropbox/thesis/VariantNET/wd/variants_chr21", 
    #                          "/home/tmsincomb/Dropbox/thesis/VariantNET/testing_data/chr21/CHROM21_v.3.3.2_highconf_noinconsistent.bed" )
    
    model = build_model(x_train, y_train)
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
from typing import Union, List, Dict, Tuple


physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('=== No GPU Detected ===')
tf.config.run_functions_eagerly(False)


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


class Models:

    def __init__(self):
        pass

    def default_model(self, input_shape: Tuple[int, int, int], **kwargs) -> keras.Model:
        inputs = Input(shape=input_shape)

        x = Conv2D(filters=16, kernel_size=(1,4), input_shape=input_shape, activation='elu', padding="same", strides=1)(inputs)
        x = MaxPool2D(pool_size=(5, 1), strides=1)(x)
        x = Conv2D(filters=32, kernel_size=(2,4), activation='elu', padding="same", strides=1)(x)
        x = MaxPool2D(pool_size=(4, 1), strides=1)(x)
        x = Conv2D(filters=48, kernel_size=(3,4), activation='elu', padding="same", strides=1)(x)
        x = MaxPool2D(pool_size=(3, 1), strides=1)(x)

        x = Flatten()(x)

        h = Dense(336, activation='elu')(x)
        dropout = Dropout(0.4)(h)
        h = Dense(168, activation='elu')(x)
        dropout = Dropout(0.2)(h)
        h = Dense(84, activation='elu')(dropout)
        dropout = Dropout(0.1)(h)
        h = Dense(42, activation='elu')(dropout)
        dropout = Dropout(0.0)(h)

        base_out = Dense(4, activation='sigmoid', name='base')(dropout)
        h = Dense(4, activation='elu')(dropout)
        genotype_out = Softmax(name='genotype')(h)

        model = Model(inputs=inputs, outputs=[base_out, genotype_out])
        
        return model 
        

    def train_model(self,
                    model: keras.Model,
                    x_train: np.array, 
                    y_train: np.array, 
                    base_loss_func: Union[str, object] = 'binary_crossentropy', 
                    genotype_loss_func: Union[str, object] = 'categorical_crossentropy',
                    optimizer: Union[str, object] = keras.optimizers.Adam(lr=0.0005),
                    loss_weights: list = [1, {0:.5, 1:.5, 2:0, 3:.5}],
                    validation_split: float = .25,
                    epochs: int = 30,
                    patience: int = 5,
                    **kwargs) -> keras.Model:
        """
        Train Keras Model

        Args:
            x_train (np.array): [description]
            y_train (np.array): [description]
            base_loss_func (Union[str, object], optional): [description]. Defaults to 'binary_crossentropy'.
            genotype_loss_func (Union[str, object], optional): [description]. Defaults to 'categorical_crossentropy'.
            optimizer (Union[str, object], optional): [description]. Defaults to keras.optimizers.Adam(lr=0.0005).
            loss_weights (list, optional): [description]. Defaults to [1, {0:.5, 1:.5, 2:0, 3:.5}].
            validation_split (float, optional): [description]. Defaults to .25.
            epochs (int, optional): [description]. Defaults to 20.
            patience (int, optional): [description]. Defaults to 1.

        Returns:
            keras.Model: [description]
        """
        optim = optimizer
        metrics = ["accuracy"]
        losses = {
            'base': base_loss_func,
            'genotype': genotype_loss_func,
        }
        model.compile(loss=losses, optimizer=optim, metrics=metrics, loss_weights=loss_weights)   
        y = {
            'base':y_train[:,:4],
            'genotype':y_train[:,4:]
        }
        early_stop = EarlyStopping(patience=patience, restore_best_weights=True)
        model.fit(x_train, y, validation_split=validation_split, epochs=epochs, callbacks=[early_stop])
        
        return model


models = Models()


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
    # utils.get_training_array("/home/tmsincomb/Dropbox/thesis/VariantNET/wd/aln_tensor_chr22", 
    #                          "/home/tmsincomb/Dropbox/thesis/VariantNET/wd/variants_chr22", 
    #                          "/home/tmsincomb/Dropbox/thesis/VariantNET/testing_data/chr22/CHROM22_v.3.3.2_highconf_noinconsistent.bed" )
    
    model = models.default_model(input_shape=x_train.shape[0])
    model = models.train_model(model, x_train, y_train)
import numpy as np
import os.path
import sys
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Flatten, Dense, Lambda
from keras import backend as K
from keras.optimizers import SGD, Adagrad, Adadelta
from keras.callbacks import CSVLogger
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

airports = ['EIDW', 'EGLL', 'LFPG', 'LFBO', 'EGPH', 'EHAM', 'EBBR', 'LEMD', 'LEBL', 'LPPT', 'LIRF',
            'LIMC', 'LSZH', 'EDDM', 'EDFH', 'EDDT', 'EKCH', 'ENGM', 'ESSA', 'EFHK', 'LOWW']

def get_rains(code):
    arr = np.load("../data/rain.npy")
    idx = airports.index(code)
    return arr[:, idx].astype(np.int32)

def get_era_full(param, level):
    arr = np.load("../data/{}{}_uint8.npy".format(param, level))
    return arr.astype(np.float32, copy=False) # / 256.

def train_model(airport):
    
    cnn_model = Sequential()
    cnn_model.add(Convolution2D(128, (3, 3), padding='same', activation='elu', name='block1_conv1', input_shape=(80,120,3)))
    cnn_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    cnn_model.add(BatchNormalization())
    cnn_model.add(Convolution2D(256, (3, 3), padding='same', activation='elu', name='block2_conv1'))
    cnn_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    cnn_model.add(Flatten())
    cnn = Lambda( lambda x: cnn_model(x) ) 

    rnn_model = Sequential()

    rnn_model.add(TimeDistributed(cnn, input_shape=(8,80,120,3)))
    rnn_model.add(LSTM(256, return_sequences=True))
    rnn_model.add(LSTM(256))
    rnn_model.add(Dense(2, activation='softmax', name='fc_cls'))

    for il, l in enumerate(rnn_model.layers):
      print il, l.name, l.output_shape

    # Import data
    params = ["z", "z", "z"]
    levels = [500, 700, 1000]

    in1_var = get_era_full(params[0], levels[0])
    in2_var = get_era_full(params[1], levels[1])
    in3_var = get_era_full(params[2], levels[2])

    x = np.concatenate((np.expand_dims(in1_var, axis=3), np.expand_dims(in2_var, axis=3), np.expand_dims(in3_var, axis=3)), axis=3)
    x -= np.mean(x, axis=(0,1,2))
    x /= np.max(np.abs(x), axis=(0,1,2))

    #13141
    X = np.zeros((13141, 8, 80, 120, 3), dtype=np.float32)
    for i in range(13141):
        #X[i,:,:,:,:] = np.rollaxis(x[i:i+8, :, :, :],0,3)
        X[i,:,:,:,:] = x[i:i+8, :, :, :]

    Y = get_rains(airport)[7:]
    b = np.zeros((Y.shape[0], 2))
    b[np.arange(Y.shape[0]), Y] = 1

    #adagrad = Adagrad(lr=0.0002)
    rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    csv_logger = CSVLogger('cnn_lstm_{}.log'.format(airport))
    rnn_model.fit(X, b, batch_size=100, epochs=300, verbose=1, validation_split=0.2, callbacks=[csv_logger])

    rnn_model.save('model_cnn_lstm_{}.h5'.format(airport))

if __name__ == '__main__':
    airport = sys.argv[1]
    train_model(airport)






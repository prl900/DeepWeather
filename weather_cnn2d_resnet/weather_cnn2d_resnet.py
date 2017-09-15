import argparse
import numpy as np
import os.path
import sys
from tensorflow.contrib.keras.python.keras import layers
from tensorflow.contrib.keras.python.keras.layers import Activation
from tensorflow.contrib.keras.python.keras.layers import AveragePooling2D
from tensorflow.contrib.keras.python.keras.layers import BatchNormalization
from tensorflow.contrib.keras.python.keras.layers import Conv2D, SeparableConv2D
from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.layers import Flatten
from tensorflow.contrib.keras.python.keras.layers import GlobalAveragePooling2D
from tensorflow.contrib.keras.python.keras.layers import GlobalMaxPooling2D
from tensorflow.contrib.keras.python.keras.layers import Input
from tensorflow.contrib.keras.python.keras.layers import MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers import ZeroPadding2D
from tensorflow.contrib.keras.python.keras.layers import Dropout
from tensorflow.contrib.keras.python.keras.models import Model
from keras import backend as K
from keras.optimizers import SGD, Adagrad, Adadelta
from keras.callbacks import CSVLogger

airports = ['EIDW', 'EGLL', 'LFPG', 'LFBO', 'EGPH', 'EHAM', 'EBBR', 'LEMD', 'LEBL', 'LPPT', 'LIRF',
            'LIMC', 'LSZH', 'EDDM', 'EDFH', 'EDDT', 'EKCH', 'ENGM', 'ESSA', 'EFHK', 'LOWW']

def get_rains(code):
    arr = np.load("../data/rain.npy")
    idx = airports.index(code)
    return arr[:, idx].astype(np.int32)

def get_era_full(param, level):
    arr = np.load("../data/{}{}_uint8.npy".format(param, level))
    return arr.astype(np.float32, copy=False) # / 256.

def identity_block(input_tensor, kernel_size, filters, stage, block):
  """The identity block is the block that has no conv layer at shortcut.

  Arguments:
      input_tensor: input tensor
      kernel_size: defualt 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filterss of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names

  Returns:
      Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if K.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  ''' resnet v1 using post-activation
  x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
  x = Activation('relu')(x)

  x = Conv2D(
      filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
  x = Activation('relu')(x)

  x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

  x = layers.add([x, input_tensor])
  x = Activation('relu')(x)
  '''

  # resnet v2 using pre-activation
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(input_tensor)
  x = Activation('elu')(x)
  x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(x)

  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
  x = Activation('elu')(x)
  x = SeparableConv2D(
      filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)

  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
  x = Activation('elu')(x)
  x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)

  x = layers.add([x, input_tensor])

  return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
  """conv_block is the block that has a conv layer at shortcut.

  Arguments:
      input_tensor: input tensor
      kernel_size: defualt 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filterss of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      strides: Tuple of integers.

  Returns:
      Output tensor for the block.

  Note that from stage 3, the first conv layer at main path is with
  strides=(2,2)
  And the shortcut should have strides=(2,2) as well
  """
  filters1, filters2, filters3 = filters
  if K.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  ''' resnet v1 using post-activation
  x = Conv2D(
      filters1, (1, 1), strides=strides,
      name=conv_name_base + '2a')(input_tensor)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
  x = Activation('relu')(x)

  x = Conv2D(
      filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
  x = Activation('relu')(x)

  x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

  shortcut = Conv2D(
      filters3, (1, 1), strides=strides,
      name=conv_name_base + '1')(input_tensor)
  shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

  x = layers.add([x, shortcut])
  x = Activation('relu')(x)
  '''

  # resnet v2 using pre-activation
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(input_tensor)
  x = Activation('elu')(x)
  x = Conv2D(
      filters1, (1, 1), strides=strides,
      name=conv_name_base + '2a')(x)

  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
  x = Activation('elu')(x)
  x = SeparableConv2D(
      filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)

  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
  x = Activation('elu')(x)
  x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)

  shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(input_tensor)
  shortcut = Activation('elu')(shortcut)
  shortcut = Conv2D(
      filters3, (1, 1), strides=strides,
      name=conv_name_base + '1')(shortcut)

  x = layers.add([x, shortcut])

  return x

def res_net(input_shape, blocks_per_stage, classes=2):
  assert len(blocks_per_stage) == 4, blocks_per_stage

  inputs = Input(shape=input_shape)

  if K.image_data_format() == 'channels_last':
    bn_axis = 3
    assert np.floor(input_shape[0] / 16.) >= 3, input_shape
    assert np.floor(input_shape[1] / 16.) >= 3, input_shape 
  else:
    bn_axis = 1
    assert np.floor(input_shape[1] / 16.) >= 3, input_shape
    assert np.floor(input_shape[2] / 16.) >= 3, input_shape

  ''' resnet v1
  x = ZeroPadding2D((3, 3))(inputs)
  x = Conv2D(64, (7, 7), name='conv1')(x)
  #x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
  #x = Activation('relu')(x)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)
  x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
  '''

  # resnet v2
  x = ZeroPadding2D((3, 3))(inputs)
  x = Conv2D(64, (7, 7), name='conv1')(x)
  x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
  x = Activation('elu')(x)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  x = conv_block(x, 3, [64, 64, 256], stage=2, block='1', strides=(1, 1))
  for i in xrange(blocks_per_stage[0]):
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=str(i+2))

  x = conv_block(x, 3, [128, 128, 512], stage=3, block='1')
  for i in xrange(blocks_per_stage[1]):
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=str(i+2))

  x = conv_block(x, 3, [256, 256, 1024], stage=4, block='1')
  for i in xrange(blocks_per_stage[2]):
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=str(i+2))

  x = conv_block(x, 3, [512, 512, 2048], stage=5, block='1')
  for i in xrange(blocks_per_stage[3]):
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block=str(i+2))
    #if i >= blocks_per_stage[3] - 3 and i < blocks_per_stage[3] - 1:
    #  x = Dropout(0.2, name='dropout%d%s' % (5, str(i+2)))(x)

  x = Activation('elu')(x)
  x = GlobalAveragePooling2D(name='avg_pool')(x)
  x = Dense(classes, activation='softmax', name='fc_cls')(x)

  total_layers = (np.sum(blocks_per_stage) + 4) * 3 + 2
  blocks_per_stage_str = ','.join([str(b+1) for b in blocks_per_stage])
  model = Model(inputs, x, name='resnet%d_%s' % (total_layers, blocks_per_stage_str))
  return model

def train_model(blocks_per_stage_str, airport):
  assert airport in airports, airports

  input_shape = (80, 120, 3)
  blocks_per_stage = [int(i) for i in blocks_per_stage_str.split(',')]

  model = res_net(input_shape, blocks_per_stage)
     
  for il, l in enumerate(model.layers):
    print il, l.name, l.output_shape

  print 'model name: %s' % model.name

  # Import data
  params = ["z", "z", "z"]
  levels = [500, 700, 1000]

  in1_var = get_era_full(params[0], levels[0])
  in2_var = get_era_full(params[1], levels[1])
  in3_var = get_era_full(params[2], levels[2])

  X = np.concatenate((np.expand_dims(in1_var, axis=3), np.expand_dims(in2_var, axis=3), np.expand_dims(in3_var, axis=3)), axis=3)
  X -= np.mean(X, axis=(0,1,2))
  X /= np.max(np.abs(X), axis=(0,1,2))

  Y = get_rains(airport)
  b = np.zeros((Y.shape[0], 2))
  b[np.arange(Y.shape[0]), Y] = 1

  #adagrad = Adagrad(lr=0.0002)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  csv_logger = CSVLogger('{}_{}.log'.format(airport, model.name))
  model.fit(X, b, batch_size=100, epochs=300, verbose=1, validation_split=0.2, callbacks=[csv_logger])

  model.save('model_{}_{}.h5'.format(airport, model.name))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('airport', help='The airport we want to train our model on', type=str)
  parser.add_argument('blocks_per_stage', help='number of residual blocks per stage (identity residual blocks only)', type=str)
  args = parser.parse_args()
  train_model(args.blocks_per_stage, args.airport)





import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from keras.layers import Input, concatenate, add, Multiply, Lambda
from keras.layers import Conv3D, MaxPooling3D, MaxPooling2D, UpSampling2D,UpSampling3D, Conv2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.models import Model
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Add, Multiply, BatchNormalization, Activation
from tensorflow.keras.models import Model

class Identity_block(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, padding = 'same'):
        super(Identity_block, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('relu')
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding,strides = strides)

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size,padding=padding,strides = 1 )

        self.shortcut = tf.keras.layers.Conv2D(filters, kernel_size = (1,1), padding=padding, strides=strides)
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        x = self.bn1(inputs)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv2(x)
        shortcut = self.shortcut(inputs)
        shortcut = self.bn3(shortcut)
        x = self.add([x, shortcut])
        return x

class Upsample_Concatenate_block(tf.keras.Model):
    def __init__(self):
        super(Upsample_Concatenate_block, self).__init__()
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs, skip):
        x = self.upsample(inputs)
        x = self.concat([x, skip])
        return x
    
class Upconv_block(tf.keras.Model):
    def __init__(self, filters, kernel_size =3, strides=2, padding = 'same'):
        super(Upconv_block, self).__init__()
        self.upconv = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')

    def call(self, inputs):
        x = self.upconv(inputs)
        x = self.bn(x)
        x = self.act(x)
        return x


class Residual_Unet(tf.keras.Model):
    def __init__(self,trainable = True, dtype=None, **kwargs):
        super(Residual_Unet, self).__init__()

        # Encoder
        self.conv1 = tf.keras.layers.Conv2D(64, 3, padding='same',strides = 1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv2D(64, 3, padding='same', strides = 1)
        self.shortcut1 = tf.keras.layers.Conv2D(64, 1, padding='same',  strides = 1)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.add1 = tf.keras.layers.Add()

        self.idblock1 = Identity_block(128, 3, 2)
        self.idblock2 = Identity_block(256, 3, 2)
        self.idblock3 = Identity_block(512, 3, 2)

        # bridge
        self.bridge = Identity_block(1024, 3, 2)

        # Decoder
        # self.up_concat1 = Upsample_Concatenate_block()
        self.Upconv1 = Upconv_block(512)
        self.up_concat1 = tf.keras.layers.Concatenate()
        self.idblock4 = Identity_block(512, 3, 1)

        # self.up_concat2 = Upsample_Concatenate_block()
        self.Upconv2 = Upconv_block(256)
        self.up_concat2 = tf.keras.layers.Concatenate()
        self.idblock5 = Identity_block(256, 3, 1)

        # self.up_concat3 = Upsample_Concatenate_block()
        self.Upconv3 = Upconv_block(128)
        self.up_concat3 = tf.keras.layers.Concatenate()
        self.idblock6 = Identity_block(128, 3, 1)

        # self.up_concat4 = Upsample_Concatenate_block()
        self.Upconv4 = Upconv_block(64)
        self.up_concat4 = tf.keras.layers.Concatenate()
        self.idblock7 = Identity_block(64, 3, 1)

        self.output_con_def = tf.keras.layers.Conv2D(2, 1, padding='same', activation='linear')


    def call(self, inputs):
        moving_image, fixed_image = inputs
        x = tf.concat([moving_image, fixed_image], axis=-1)

        # Encoder
        e0 = self.conv1(x)
        e0 = self.bn1(e0)
        e0 = self.act1(e0)
        e0 = self.conv2(e0)
        shortcut1 = self.shortcut1(e0)
        e0 = self.bn2(shortcut1)
        e0 = self.add1([e0, shortcut1])

        e1 = self.idblock1(e0)
        e2 = self.idblock2(e1)
        e3 = self.idblock3(e2)

        # bridge
        bridge = self.bridge(e3)

        # Decoder
        # d1 = self.up_concat1(bridge, e3)
        d1 = self.Upconv1(bridge)
        d1 = Concatenate()([d1, e3])
        d1 = self.idblock4(d1)

        # d2 = self.up_concat2(d1, e2)
        d2 = self.Upconv2(d1)
        d2 = Concatenate()([d2, e2])
        d2 = self.idblock5(d2)

        # d3 = self.up_concat3(d2, e1)
        d3 = self.Upconv3(d2)

        d3 = Concatenate()([d3, e1])
        d3 = self.idblock6(d3)

        # d4 = self.up_concat4(d3, e0)
        d4 = self.Upconv4(d3)
        d4 = Concatenate()([d4, e0])
        d4 = self.idblock7(d4)

        output_def = self.output_con_def(d4)


        return output_def
    
    def get_config(self):
        config = super().get_config()
        return config




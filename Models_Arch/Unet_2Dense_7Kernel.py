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
class Conv_block(tf.keras.Model):
    def __init__(self,num_filters, kernel_size = 3):
        super(Conv_block, self).__init__()
        self.conv1 = Conv2D(num_filters, kernel_size, padding = 'same', kernel_initializer = 'he_normal')
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        self.conv2 = Conv2D(num_filters, kernel_size, padding = 'same', kernel_initializer = 'he_normal')
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

class UpConv_block(tf.keras.Model):
    def __init__(self, num_filters):
        super(UpConv_block, self).__init__()
        self.upconv = Conv2DTranspose(num_filters, 3, strides = 2, padding = 'same')
        self.bn = BatchNormalization()
        self.act = Activation('relu')

    def call(self, inputs):
        x = self.upconv(inputs)
        x = self.bn(x)
        x = self.act(x)
        return x

class Max_pool(tf.keras.Model):
    def __init__(self):
        super(Max_pool, self).__init__()
        self.pool = MaxPooling2D(pool_size = (2,2))

    def call(self, inputs):
        x = self.pool(inputs)
        return x

class Unet_2Dense_7Kernel(tf.keras.Model):
    def __init__(self,trainable = True, dtype=None, **kwargs):
        super(Unet_2Dense_7Kernel, self).__init__()
        self.conv_block1 = Conv_block(64,7)
        self.pool1 = Max_pool()
        self.conv_block2 = Conv_block(128,5)
        self.pool2 = Max_pool()
        self.conv_block3 = Conv_block(256,3)
        self.pool3 = Max_pool()
        self.conv_block4 = Conv_block(512,3)
        self.pool4 = Max_pool()

        # bottleneck
        self.conv_block5 = Conv_block(1024,3)

        self.upconv_block1 = UpConv_block(512)
        self.conv_block6 = Conv_block(512)
        self.upconv_block2 = UpConv_block(256)
        self.conv_block7 = Conv_block(256)
        self.upconv_block3 = UpConv_block(128)
        self.conv_block8 = Conv_block(128)
        self.upconv_block4 = UpConv_block(64)
        self.conv_block9 = Conv_block(64)

        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        

        self.output_def = Conv2D(2, 1, activation = 'linear')

    def call(self, inputs):
        moving, fixed = inputs
        inputs = concatenate([moving, fixed], axis = -1)

        conv1 = self.conv_block1(inputs)
        pool1 = self.pool1(conv1)
        conv2 = self.conv_block2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv_block3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv_block4(pool3)
        pool4 = self.pool4(conv4)

        # bottleneck
        conv5 = self.conv_block5(pool4)

        upconv1 = self.upconv_block1(conv5)
        concat1 = Concatenate()([conv4, upconv1])
        conv6 = self.conv_block6(concat1)
        upconv2 = self.upconv_block2(conv6)
        concat2 = Concatenate()([conv3, upconv2])
        conv7 = self.conv_block7(concat2)
        upconv3 = self.upconv_block3(conv7)
        concat3 = Concatenate()([conv2, upconv3])
        conv8 = self.conv_block8(concat3)
        upconv4 = self.upconv_block4(conv8)
        concat4 = Concatenate()([conv1, upconv4])
        conv9 = self.conv_block9(concat4)

        output = self.output_def(conv9)
        return output



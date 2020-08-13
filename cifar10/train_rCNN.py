# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:08:39 2020

@author: Jay
"""

''' 
Standard LP_64:
    No random adv perturbations
'''

import keras
from tf_agumentation import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import regularizers
import setup_data
import tensorflow as tf

#%%
def loadParam(layerDir):
    l=0
    print('loading params...'+str(l))
    means = np.load(layerDir + "mean3x3.npy")
    convs = np.load(layerDir + "covs3x3.npy")
    
    means = means.astype(np.float32)
    convs = convs.astype(np.float32)
    return means, convs 

def get_divider(_shape):
    tf_inp_ones = tf.ones((1, _shape-2, _shape-2, 1))
    tf_ones_2 = tf.ones((3,3,1,1)) 
    out2 = tf.nn.conv2d_transpose(tf_inp_ones, tf_ones_2, output_shape= [1, _shape, _shape,1], strides=(1,1,1,1), padding="VALID")
    sess = tf.Session()

    with sess.as_default():
        np_divider = out2.eval(feed_dict={})
        np_divider = np.transpose(np_divider, (1,2,3,0))
        return np_divider

def reconstruct(tf_x, out, means, tf_divider):
    out = tf.pow(2.0, out*25)
    out = tf.nn.relu(out)+1e-10
    out = out / tf.expand_dims(tf.reduce_sum(out, [3]), axis=-1)
    input_shape = tf.shape(tf_x)
    out = tf.nn.conv2d_transpose(out, means, output_shape= input_shape, strides=(1,1,1,1), padding="VALID")
    
    out = tf.transpose(out, (1,2,3,0))
    out = out/tf_divider
    out = tf.transpose(out, (3,0,1,2))        
    return out

def rbf_layer(tf_x, means, covs, divider):
    ch = means.shape[2]
    tf_ones = tf.ones((3,3,ch,1))    
    tensor1 = tf.nn.conv2d(tf_x, means, strides=(1,1,1,1), padding="VALID")
    tensor2 = tf.multiply(tf_x, tf_x, name=None)
    tensor2 = tf.nn.conv2d(tensor2, tf_ones, strides=(1,1,1,1), padding="VALID")    
    tensor3 = tf.multiply(means, means, name=None)
    tensor3 = tf.reduce_sum(tensor3, [0,1,2]) 
    out = 2*tensor1 - tensor2 - tensor3
    out = out/(2*tf.exp(covs)+ 1e-25) - covs*0.5
    out = tf.nn.sigmoid(out)
    
    return reconstruct(tf_x, out, means, divider)

#%%
class cifar10vgg:
    def __init__(self):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]
        self.model = self.build_model()
        
#%%
    def build_model(self, layer_path = './layer/lp_64/'):
        
        dr_rate = 0.35
        f = [96, 192, 384, 512, 640]
        model = Sequential()
        weight_decay = self.weight_decay
        
        means, convs = loadParam(layerDir=layer_path)
        print(means.shape)
        print(convs.shape)
        _args = {'means':means, 'covs': convs, 'divider': get_divider(self.x_shape[1])}            

        model.add(Lambda(rbf_layer, arguments=_args,  output_shape=self.x_shape, input_shape=self.x_shape)) 
        model.add(BatchNormalization())
        
        model.add(Conv2D(f[0], (3, 3), padding='same', input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dr_rate))

        model.add(Conv2D(f[0], (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(f[1], (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dr_rate))

        model.add(Conv2D(f[1], (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(f[2], (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dr_rate))

        model.add(Conv2D(f[2], (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dr_rate))

        model.add(Conv2D(f[2], (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(f[3], (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dr_rate))

        model.add(Conv2D(f[3], (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dr_rate))

        model.add(Conv2D(f[3], (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(f[4], (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dr_rate))

        model.add(Conv2D(f[4], (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dr_rate))

        model.add(Conv2D(f[4], (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dr_rate))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dense(self.num_classes))
#        model.summary()
        model.compile(loss=fn, optimizer='sgd', metrics=['accuracy'])
        return model

#%%
    def train(self, model, data, model_path, existing_path = None):
        
        label_smooth = 0.1
        y_train = data.train_labels .clip(label_smooth / 9., 1. - label_smooth)

        #training parameters
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.001     # 0.001, 0.0001
        lr_drop = 20               # init: 20, 20, 40
        
        #Epochs: 100+ 80 + (100)
        def lr_scheduler(epoch):
            epoch += 50
            lr = learning_rate * (0.5 ** (epoch // lr_drop))
            if lr < 1e-6:
                lr = 1e-6
            return lr
        
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        #data augmentation
        datagen = ImageDataGenerator(
                horizontal_flip=True,
                fill_mode= 'constant')
        
        datagen.fit(data.train_data)

        model_checkpoint= keras.callbacks.ModelCheckpoint(
                model_path, monitor="val_acc", save_best_only= False,
                save_weights_only=True, verbose=1)
        
        if existing_path !=None:
            model.load_weights(existing_path)

        opt = optimizers.Adam(lr=learning_rate)
        model.compile(loss=fn, optimizer=opt, metrics=['accuracy'])
        
        # original images
        train_gen = datagen.flow(data.train_data, y_train, batch_size=batch_size)
        
        def flow_noisy(train_gen):
            EPSILON = np.random.uniform(0.005, 0.03)
            
            while True:
                _, x0, y  = train_gen.next()
                
                lower = np.clip(x0-EPSILON, 0, 1)
                upper = np.clip(x0+EPSILON, 0, 1)

                # noisy data gen
                in_rand = np.random.normal(0,0.03,[x0.shape[0], 32, 32, 3]).astype(np.float32)
                x1 = x0 + in_rand
                x1 = np.clip(x1, lower, upper)
                
                xf = np.concatenate((x0, x1), axis=0)
                yf = np.concatenate((y, y), axis=0)        
                yield (xf, yf)


        model.fit_generator(flow_noisy(train_gen),
                            steps_per_epoch=int(np.ceil(data.train_data.shape[0] / float(batch_size))),
                            epochs=maxepoches, validation_data=(data.test_data, data.test_labels), 
                            callbacks=[reduce_lr, model_checkpoint], verbose=1)

        return model
#%%
def fn(correct, predicted):
    ceLoss = tf.nn.softmax_cross_entropy_with_logits(labels=correct, 
                                                     logits=predicted)
    
    uni_in = tf.ones_like(correct) *0.01
    smoothLoss = tf.nn.softmax_cross_entropy_with_logits(labels=uni_in, 
                                                         logits=predicted)
    
    return ceLoss + 10*smoothLoss

#%%
if __name__ == '__main__':
    import os
    os.environ['PYTHONHASHSEED'] = '10'
    tf.set_random_seed(6789)
    np.random.seed(9988)
    
    data = setup_data.Cifar()
    print(data.train_data.shape, data.test_data.shape)
    model = cifar10vgg()
    if not os.path.exists('./models/'):
        os.makedirs('./models/')
        
    model_path = './models/rCNN'
    vgg = model.train(model= model.model, data= data, model_path= model_path, existing_path= None)
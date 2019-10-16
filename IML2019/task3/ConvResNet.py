# -*- coding: utf-8 -*-
import keras 
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input, add

def cook():
        
    def relu(x):
        return Activation('relu')(x)
    
    def neck(nip,nop,stride):
        def unit(x):
            nBottleneckPlane = int(nop / 4)
            nbp = nBottleneckPlane
    
            if nip==nop:
                ident = x
    
                x = BatchNormalization(axis=-1)(x)
                x = relu(x)
                x = Convolution2D(nbp,1,1,
                subsample=(stride,stride))(x)
    
                x = BatchNormalization(axis=-1)(x)
                x = relu(x)
                x = Convolution2D(nbp,3,3,border_mode='same')(x)
    
                x = BatchNormalization(axis=-1)(x)
                x = relu(x)
                x = Convolution2D(nop,1,1)(x)
    
                out = merge([ident,x],mode='sum')
            else:
                x = BatchNormalization(axis=-1)(x)
                x = relu(x)
                ident = x
    
                x = Convolution2D(nbp,1,1,
                subsample=(stride,stride))(x)
    
                x = BatchNormalization(axis=-1)(x)
                x = relu(x)
                x = Convolution2D(nbp,3,3,border_mode='same')(x)
    
                x = BatchNormalization(axis=-1)(x)
                x = relu(x)
                x = Convolution2D(nop,1,1)(x)
    
                ident = Convolution2D(nop,1,1,
                subsample=(stride,stride))(ident)
    
                out = merge([ident,x],mode='sum')
    
            return out
        return unit
    
    def cake(nip,nop,layers,std):
        def unit(x):
            for i in range(layers):
                if i==0:
                    x = neck(nip,nop,std)(x)
                else:
                    x = neck(nop,nop,1)(x)
            return x
        return unit
    
    inp = Input(shape=(32,32,3))
    i = inp
    
    i = Convolution2D(16,3,3,border_mode='same')(i)
    
    i = cake(16,32,3,1)(i) #32x32
    i = cake(32,64,3,2)(i) #16x16
    i = cake(64,128,3,2)(i) #8x8
    
    i = BatchNormalization(axis=-1)(i)
    i = relu(i)
    
    i = AveragePooling2D(pool_size=(8,8),border_mode='valid')(i) #1x1
    i = Flatten()(i) # 128
    
    i = Dense(10)(i)
    i = Activation('softmax')(i)
    
    model = Model(input=inp,output=i)

return model
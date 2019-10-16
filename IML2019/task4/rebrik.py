# -*- coding: utf-8 -*-


import keras
import tensorflow as tf
from keras.layers import Layer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
#%%

class addGamma( Layer ):
    # Beta parameters
    def __init__(self  , **kwargs):
        super(addGamma, self).__init__(**kwargs)
        
    def build(self, input_shape):
        if self.built:
            return
        self.gamma = self.add_weight(name='gamma', shape= input_shape[1:], initializer='ones', trainable=True)
        self.built = True
        super(addGamma, self).build(input_shape)  
        
    def call(self, x , training=None):
        return tf.add( x , self.gamma )

class denoisingGauss( Layer ):
    # Gaussian denoising function
    def __init__(self , **kwargs):
        super(denoisingGauss, self).__init__(**kwargs)
        
    def wi(self ,  init , name):
        if init == 1:
            return self.add_weight(name='guess_'+name, shape=( self.size, ), initializer='ones', trainable=True)
        elif init == 0:
            return self.add_weight(name='guess_'+name, shape=( self.size, ), initializer='zeros', trainable=True)
        
    def build(self, input_shape):
        self.size = input_shape[0][-1]
        self.a1 = self.wi(0., 'a1')
        self.a2 = self.wi(1., 'a2')
        self.a3 = self.wi(0., 'a3')
        self.a4 = self.wi(0., 'a4')
        self.a5 = self.wi(0., 'a5')
        self.a6 = self.wi(0., 'a6')
        self.a7 = self.wi(1., 'a7')
        self.a8 = self.wi(0., 'a8')
        self.a9 = self.wi(0., 'a9')
        self.a10 = self.wi(0., 'a10')
        super( denoisingGauss , self).build(input_shape)    
        
    def call(self, x):
        z_c, u = x 
        a1 = self.a1 
        a2 = self.a2 
        a3 = self.a3 
        a4 = self.a4 
        a5 = self.a5 
        a6 = self.a6 
        a7 = self.a7 
        a8 = self.a8 
        a9 = self.a9 
        a10 =self.a10
        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10
        z_est = (z_c - mu) * v + mu
        return z_est
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.size)

def batchNorm(batch, mean=None, var=None):
    if mean is None or var is None:
        mean, var = tf.nn.moments(batch, axes=[0])
    return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

def getLadderNetwork(  ):
    from keras.models import Model
    from keras.layers import Input, Dense, GaussianNoise, Activation, Lambda
    
    layer_sizes =    [139, 512, 512, 256, 10]
    denoising_cost = [1000, 10, .10, .10, .10] 
    
    noise_std = 0.3
    L = len(layer_sizes) - 1 
    
    inputs_labeled = Input((layer_sizes[0],))  
    inputs_unlabeled = Input((layer_sizes[0],))  
    dense_encoder = [ Dense( size , use_bias=False , kernel_initializer='glorot_normal')  for size in layer_sizes[1:]    ]
    dense_decoder = [ Dense( size , use_bias=False , kernel_initializer='glorot_normal')  for size in layer_sizes[:-1]  ]
    gammas =[ addGamma() for l in range(L)]    
    
    def encoder( inputs, noise_std ):
        h = GaussianNoise(noise_std)(inputs)
        latent_z = [ None for _ in range( len(layer_sizes)  )]
        latent_z[0] = h
        
        for l in range(1, L+1):
            z_preBN = dense_encoder[l-1](h)
            z_preNoise = Lambda(batchNorm)(z_preBN)
            z = GaussianNoise(noise_std)(z_preNoise)
            if l == L:  h = Activation('softmax')(gammas[l-1](z))
            else:       h = Activation('relu')(gammas[l-1](z)) 
            latent_z[l] = z

        return h  , latent_z
        
    y_corrupted_labeled , _   = encoder(inputs_labeled, noise_std)
    y_labeled  , _  = encoder(inputs_labeled, 0.0) 

    y_corrupted_unlabeled , corrupted_z  = encoder(inputs_unlabeled, noise_std)
    y_unlabeled ,  clean_z = encoder(inputs_unlabeled, 0.0 )  
    
    # Decoder
    decoder_cost = []
    for l in range(L,-1,-1):
        z = clean_z[l]
        z_tilda = corrupted_z[l]
        if l == L:   u_preBN = y_corrupted_unlabeled
        else:        u_preBN = dense_decoder[l](z_est)
        u = Lambda(batchNorm)(u_preBN)
        z_est = denoisingGauss()([z_tilda,u])
        decoder_cost.append(((tf.reduce_sum(tf.square(z_est  - z), 1)) / layer_sizes[l]) * denoising_cost[l])
        
        
    Cost_unsupervised = tf.add_n(decoder_cost)
    
    output = Lambda(lambda x: x[0])([y_corrupted_labeled, 
                                     y_labeled,
                                     y_corrupted_unlabeled,
                                     y_unlabeled,
                                     u, z_est, 
                                     z
                                     ])
    
    lr_train = Model(inputs=[inputs_labeled,inputs_unlabeled],outputs=output)
    lr_train.add_loss( Cost_unsupervised )
    lr_train.compile( optimizer=keras.optimizers.Adam(lr=0.001) , loss='categorical_crossentropy', metrics=['accuracy'])
    lr_train.metrics_names.append("loss_u")
    lr_train.metrics_tensors.append(Cost_unsupervised)

    lr_test = Model( inputs_labeled , y_labeled  )
    lr_train.test_model = lr_test
    return lr_train
#%%
my_model = getLadderNetwork()
keras.utils.vis_utils.plot_model(my_model, to_file='archtest.png', show_shapes=True, show_layer_names=True)

#model = get_ladder_network_fc(  layer_sizes = [139, 1024, 512, 256, 10] , 
#                                noise_std = 0.3  ,
#                                denoising_cost = [100.0, 10.0, 0.10, 0.10, 0.10] )


#%%
    

train_labeled = pd.read_csv('train_labeled.csv')
train_unlabeled = pd.read_csv('train_unlabeled.csv')
test = pd.read_csv('test.csv')

train_labels = pd.DataFrame(train_labeled['y'])
train_labeled = train_labeled.drop(['y'], axis=1)

# Data Information
ntrain = train_labeled.shape[0]
ntrain_unlabeled = train_unlabeled.shape[0]
ntest = test.shape[0]
nfeatures = train_labeled.shape[1]
nclasses = len(np.unique(train_labels['y']))
nweights = ntrain / (nclasses * np.bincount(train_labels['y']))


def InitializeData(train=train_labeled, train_labels=train_labels, test=test, unlabeled = train_unlabeled):  
    x_train = train.values 
    x_test = test.values  
    y_train = train_labels.values.ravel()
    x_unlabeled = unlabeled.values
    # Scale data
    if True :  
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)  
        x_unlabeled = scaler.fit_transform(x_unlabeled)
    return x_train,y_train,x_test,x_unlabeled



x_train, y_train, x_test, x_unlabeled = InitializeData()
nvalidation = 2/9

x_train, x_valid, y_train, y_valid = train_test_split(
           x_train, y_train, test_size = nvalidation, random_state=1, stratify = y_train)

y_train_mat = to_categorical(y_train, num_classes=nclasses)
y_valid_mat = to_categorical(y_valid, num_classes=nclasses) 


x_train_labeled = x_train
x_train_unlabeled = x_unlabeled
y_train_labeled = y_train_mat
rep = int(x_train_unlabeled.shape[0] / x_train_labeled.shape[0])
x_train_labeled_rep = np.concatenate([x_train_labeled]*rep)
y_train_labeled_rep = np.concatenate([y_train_labeled]*rep)


#%%
nepochs=200
nbatchsize=25
validation = np.zeros(nepochs)
for i in range(nepochs):
    print('Epoch:',i+1)
    my_model.fit([x_train_labeled_rep,x_train_unlabeled] , y_train_labeled_rep, 
                 epochs=1, batch_size=nbatchsize , shuffle=True)
    y_valid_prob = my_model.test_model.predict(x_valid , batch_size=100 )
    validation[i] = accuracy_score(y_valid_mat.argmax(-1) , y_valid_prob.argmax(-1) )
    print("validation accuracy: " , validation[i] )
  
plt.plot(validation)  
plt.show  

 #%%

x_valid_labeled_rep = np.concatenate([np.concatenate([x_train,x_valid]*2),x_valid,x_valid[:1000,]])
y_valid_labeled_rep = np.concatenate([np.concatenate([y_train_mat,y_valid_mat]*2),y_valid_mat,y_valid_mat[:1000,]])

nepochs=50

for i in range(nepochs):
    print('Extra Epoch:',i+1)
    
    my_model.fit([x_valid_labeled_rep,x_train_unlabeled],y_valid_labeled_rep,
                  epochs=1, batch_size=nbatchsize , shuffle=True)
    y_test_prob = my_model.test_model.predict(x_test , batch_size=100 )
    y_test_hat = y_test_prob.argmax(-1)
    
def save_solution(csv_file,solution):
    with open(csv_file, 'w') as csv:
        df = pd.DataFrame.from_dict({'Id':range(ntrain+ntrain_unlabeled,ntrain+ntrain_unlabeled+len(solution)),'y': solution})
        df.to_csv(csv,index = False)
        
save_solution('solution.csv', y_test_hat )
save_solution('LN_probabilities.csv',y_test_prob.max(1))  

#%%
#model = get_ladder_network_fc(  layer_sizes = [139, 1024, 512, 256, 10] , 
#                                noise_std = 0.3  ,
#                                denoising_cost = [100.0, 10.0, 0.10, 0.10, 0.10] )


'''
nepochs=100
for i in range(nepochs):
    print('Epoch:',i+1)
    nbatchsize=25
    model.fit([x_train_labeled_rep, x_train_unlabeled] , y_train_labeled_rep, 
              epochs=1, batch_size=nbatchsize )
    y_test_prob = model.test_model.predict(x_test , batch_size=100 )
    y_test_hat = y_test_prob.argmax(-1)
        
    
def save_solution(csv_file,solution):
    with open(csv_file, 'w') as csv:
        df = pd.DataFrame.from_dict({'Id':range(ntrain+ntrain_unlabeled,ntrain+ntrain_unlabeled+len(solution)),'y': solution})
        df.to_csv(csv,index = False)
        
save_solution('solution.csv', y_test_hat )    

'''
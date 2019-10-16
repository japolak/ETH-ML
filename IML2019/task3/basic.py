# -*- coding: utf-8 -*-

from keras.layers import Input, Dense
from keras.models import Model

#%%
# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

plot_model(model, to_file='basic_plot.png', show_shapes=True, show_layer_names=True)
#model.fit(data, labels)  # starts training
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf


# In[ ]:


tf.__version__


# In[ ]:


tf.keras.__version__


# Keras is a high level API for tensorflow

# In[ ]:


#from tensorflow import keras

#Earlier Keras and tensorflow were seperate , but in tf version 2 keras has been adopted inside tf, so now we can use them both with tf.


# In[ ]:


tf.config.list_physical_devices("GPU")


# In[ ]:


tf.config.list_physical_devices("CPU")


# In[ ]:


# We will MNIST dataset - Containing images of very less size - 28*28 pixels of diff digits or letter

# Though CNN is used for images , but since we have very less size resolution data , ANN can be use.


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ### Working on mnist dataset - 
# 
# * This dataset contains handwritten digits. 
# * It has 10 classes i.e. 0 to 9
# * Each data point is 2D array of 28x28 size.
# * Also known as hello world dataset for ANN
# 
# [image source](https://en.wikipedia.org/wiki/MNIST_database#/media/File:MnistExamples.png)
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

# In[ ]:


# Mnist dataset is already available in tf
mnist = tf.keras.datasets.mnist


# In[ ]:


#Now we have to load it before using
#mnist.load_data()
(X_train_full, y_train_full),(X_test, y_test) = mnist.load_data()


# In[ ]:


X_train_full.shape


# In[ ]:


y_train_full.shape


# In[ ]:


X_train_full[0].shape # Size of first image


# In[ ]:


X_train_full[0]


# In[ ]:


X_train_full


# In[ ]:


# We could see the actual image 
# So for that
img = X_train_full[0]
plt.imshow(img)
plt.show()


# In[ ]:


# We could see the actual image 
# So for that
img = X_train_full[0]
plt.imshow(img, cmap='binary')
plt.show()


# In[ ]:


# We could see the actual image 
# So for that
img = X_train_full[0]
plt.imshow(img, cmap='binary')
plt.axis("off") # to remove the X & y axis
plt.show()


# In[ ]:





# In[ ]:


y_train_full.shape


# In[ ]:


y_train_full[0]


# In[ ]:


## So we can see that the first record in X_train_full with the pixels shows digit 5


# In[ ]:


# To see that image 
plt.figure(figsize=(20,20))
sns.heatmap(img, annot=True, cmap='binary')


# In[ ]:


# To see that image 
plt.figure(figsize=(20,20))
sns.heatmap(img/255, annot=True, cmap='binary'); # img/255 as we can see above the values are in far exponenents and thus to scale them down b/w 0&1


# In[ ]:


y_train_full.shape


# In[ ]:


# data split 
# And also normalising them to be between 0&1 by dividing by 255
# So out of 60k first 5k will be validation and rest 55k will be used for our training purposes
X_valid, X_train = X_train_full[:5000]/255., X_train_full[5000:]/255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

X_test = X_test/255.


# In[ ]:


# We will create a list of all the layers

LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
          tf.keras.layers.Dense(units=300,activation="relu", name="HiddenLayer1"),
          tf.keras.layers.Dense(units=100,activation="relu", name="HiddelLayer2"),
          tf.keras.layers.Dense(units=10,activation="softmax", name="OutputLayer")
]


# In[ ]:


# Creating model
model_clf = tf.keras.models.Sequential(LAYERS)


# In[ ]:


model_clf.layers


# In[ ]:


model_clf.summary()


# In[ ]:


# First layer * second layer + bias

784*300+300, 300*100+100, 100*10+10


# In[ ]:


np.sum((235500, 30100, 1010))


# In[ ]:


# So that is how 266610 traininable parameters come from
# So for every layer the input * weights + bias is shown above


# In[ ]:





# In[ ]:


model_clf.layers[0].name


# In[ ]:


model_clf.layers[1].name


# In[ ]:


model_clf.layers[1].get_weights()


# In[ ]:


weights, biases = model_clf.layers[1].get_weights()


# In[ ]:


weights.shape


# In[ ]:


biases.shape


# In[ ]:





# In[ ]:


# Now we need few things
LOSS_FUNCTION = "sparse_categorical_crossentropy"
OPTIMIZER = "SGD" # Stochastic gradient descent
METRICS = ['accuracy']

model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)


# ### Why 1719 data points at each epoch
# By default the batch_size=32.
# 
# Since of total 60k data points 55k as training, so at each epoch the data points will be:-
# 
# 55000/32 = 1718.75 = 1719.
# 
# We can change the batch size.[link text](https://)

# In[ ]:


# model compiled

# Now we need to train
EPOCHS = 30
VALIDATION = (X_valid, y_valid)

history = model_clf.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION)


# In[ ]:


history # is an object


# In[ ]:


history.params


# In[ ]:


history.history


# In[ ]:


pd.DataFrame(history.history)


# In[ ]:


pd.DataFrame(history.history).plot(figsize=(10,7))
plt.grid(True)
plt.show()


# In[ ]:


# Let's evaluate
model_clf.evaluate(X_test, y_test)


# ### Why 313 data points at each epoch
# By default the batch_size=32.
# 
# Since of total 10k data points are in test, so at each epoch the data points will be:-
# 
# 10000/32 = 312.5 = 313
# 
# We can change the batch size.

# In[ ]:





# In[ ]:


# let's predict
X_new = X_test[:3] # taking only firt 3 data points out of 10k test data ponnts

y_prob = model_clf.predict(X_new)
y_prob


# In[ ]:


# let's predict
X_new = X_test[:3]

y_prob = model_clf.predict(X_new)
y_prob.round(2)


# In[ ]:


# So the predict gives out probability of the each data point to belonging to a class amoung the 10 numbers form 0-9, i.e. the data points is which no. among 0-9


# In[ ]:


y_prob.shape


# In[ ]:


y_prob


# In[ ]:


# TO get the actual class/digit prediction 
y_pred = np.argmax(y_prob)
y_pred


# In[ ]:


# TO get the actual class/digit prediction 
y_pred = np.argmax(y_prob, axis=-1)
y_pred


# In[ ]:


# SO the model is predicting the first 3 data points as 7,2,1.


# In[ ]:





# In[ ]:


# Plotting
for img_array, pred, actual in zip(X_new, y_pred, y_test[:3]):
  plt.imshow(img_array, cmap="binary")
  plt.title(f"predicted: {pred}, Actual: {actual}")
  plt.axis("off")
  plt.show()
  print("---"*20)


# In[ ]:


# saving the model
model_clf.save("model.h5")


# In[ ]:





# In[ ]:





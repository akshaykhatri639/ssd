
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras
from keras.applications import vgg19
from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D, Input
from keras.models import Model
from functools import partial, update_wrapper
from DataGenerator import DataGenerator
# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')
from keras import optimizers
from keras.metrics import sparse_categorical_accuracy, categorical_accuracy
from keras import backend as K

from losses_and_metrics import accuracy, recall, loss_with_negative_mining, wrapped_partial, compute_one_by_N


# In[2]:


num_classes = 21
aspect_ratios = [1, 2, 3, 1 / 2.0, 1 / 3.0]
num_aspect_ratios = len(aspect_ratios)+1 # +1 for the last box with aspect_ratio 1 but bigger size

# feature_sizes = [28, 14, 7]
feature_sizes = [28, 14]

batch_size = 32


# In[3]:


# Use VGG as the base model
model = vgg19.VGG19(include_top=False, input_shape=(224, 224, 3))


# In[4]:


# see output shapes on all layers
for layer in model.layers:
    print layer, layer.output_shape
    layer.trainable = True


# In[5]:


out1 = Conv2D(padding='same', filters=num_classes*num_aspect_ratios, kernel_size=3,
              activation=None, name='28')(model.layers[-7].output)

out2 = Conv2D(padding='same', filters=num_classes*num_aspect_ratios, kernel_size=3, 
              activation=None, name='14')(model.layers[-2].output)

# out3 = Conv2D(padding='same', filters=num_classes*num_aspect_ratios, kernel_size=3,
#               activation=None, name='7')(model.layers[-1].output)


# In[6]:


print out1.shape
print out2.shape
# print out3.shape


# In[7]:


k = 20 # set randomly for now


# In[12]:


ssd_model = Model(inputs=model.input, outputs = [out1, out2])
acc_fun = wrapped_partial(accuracy, num_aspect_ratios=num_aspect_ratios, num_classes=num_classes)
recall_fun = wrapped_partial(recall, num_aspect_ratios=num_aspect_ratios, num_classes=num_classes)

loss_fun_28 = wrapped_partial(loss_with_negative_mining, k=13, num_aspect_ratios=num_aspect_ratios, num_classes=num_classes)
loss_fun_14 = wrapped_partial(loss_with_negative_mining, k=40, num_aspect_ratios=num_aspect_ratios, num_classes=num_classes)

optim = optimizers.Adam()

ssd_model.compile(optimizer=optim, 
              loss={'28': loss_fun_28, '14': loss_fun_14}, #, '7':loss_fun},
                 metrics=[acc_fun, recall_fun])

# ssd_model.load_weights("VGG_basic")


# In[13]:


data_gen = DataGenerator(data_dir='../data/VOCdevkit/VOC2012/JPEGImages/', 
                        label_dir='../data/VOCdevkit/VOC2012/Preprocessed/', 
                        num_classes=num_classes, num_aspect_ratios=num_aspect_ratios,
                        feature_sizes=feature_sizes, 
                        batch_size=batch_size)


# In[14]:


# t = data_gen.generate()
# print t.next()


# In[ ]:


ssd_model.fit_generator(generator=data_gen.generate(),steps_per_epoch=1000, epochs=20, 
                       validation_data=data_gen.generate(train=False), validation_steps=32)


# In[ ]:


ssd_model.save("VGG_basic_1")


# In[ ]:


# saffa


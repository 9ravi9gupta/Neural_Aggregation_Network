
# coding: utf-8

# In[11]:


from google.colab import drive
drive.mount('./drive')


# ## Data PreProcessing 

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import os
from keras.utils import to_categorical


# In[ ]:


X_train = np.load('drive/My Drive/dataset/X_train.npy')
Y_train = np.load('drive/My Drive/dataset/y_train.npy')


# In[ ]:


os.mkdir('images')  #--making directory for storing the reshaped images
 
for i in range(X_train.shape[0]):
    img = X_train[i].reshape(50,37)
    filename = 'images'+'/train_Img_'+ str(i+1) + '.png'
    plt.imsave(filename,img)
    
    
#-- changing the dimensions of images from 50x37 to 224x224
train_imgs = os.listdir('images')
reshaped_train_imgs = []

for img_name in train_imgs:
    file_dir = 'images/' + img_name
    img = image.load_img(file_dir,target_size=(224,224))
    img = image.img_to_array(img)
    reshaped_train_imgs.append(img)


# In[17]:


reshaped_train_imgs[0].shape


# In[ ]:


train_x = np.stack(reshaped_train_imgs)  #--stacking up all images into batches
train_y = to_categorical(Y_train)


# In[19]:


print(train_x.shape)
print(train_y.shape)


# ## Training 

# In[ ]:


from keras.layers import Dense,Flatten,Input,Conv2D,BatchNormalization,MaxPool2D
from keras.models import Sequential


# In[16]:


model = Sequential()

model.add(Conv2D(filters=64,kernel_size=[3,3],activation='relu',input_shape=(224,224,3)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(BatchNormalization())
model.add(Conv2D(filters=64,kernel_size=[3,3],activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(BatchNormalization())
model.add(Conv2D(filters=64,kernel_size=[3,3],activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(BatchNormalization())
model.add(Conv2D(filters=64,kernel_size=[3,3],activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(BatchNormalization())
model.add(Flatten())

model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=7,activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[18]:


history = model.fit(x=train_x, y=train_y, batch_size=64, epochs=10)


# In[ ]:


labels = Y_train.reshape(len(Y_train),1)


# In[22]:


from matplotlib import pyplot

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label=labels[:,0])

# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()


# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['acc'], label=labels[:,0])

# pyplot.plot(history.history['val_acc'], label='test')
# pyplot.legend()
pyplot.show()


# ## Saving Weights and Architecture of trained Model

# In[ ]:


model.save_weights('drive/My Drive/dataset/model_weights.hdf5')

model_yaml = model.to_yaml()  #--Serialize the Yaml

with open('drive/My Drive/dataset/model_arch.yaml','w') as yaml_file:  #--saving the architecture of model
    yaml_file.write(model_yaml)


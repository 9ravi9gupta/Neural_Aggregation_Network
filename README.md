# What is Neural Aggregation Network(NAN)?

The network which takes a face video or face image set of a person with a
variable number of face images as its input, and produces a compact, fixeddimension
feature representation for recognition.
Neural Aggregation Network has two modules:‐
      
      • Feature Embedding Module
      • Aggregation Module
      
      
  ## Feature Embedding Module:‐

The feature embedding module is a deep Convolutional Neural Network
(CNN) which maps each face image to a feature vector.

It takes set of images as a input, and after number of convolutional over the images

it give a feature vector for that set of images which is further passes to aggregation
module.

FE Module uses Pre‐trained Module like VGG16, GoogleLeNet to obtain the feature
vectors of given set of images.

Feature vectors is passed to aggregation module for further operations. The
GoogLeNet produces 128‐dimension image features, which are first normalized to
be unit vectors then fed into the aggregation module.



## Aggregation Module:‐

The aggregation module consists of two attention blocks which adaptively
aggregate the feature vectors to form a single feature inside the convex hull
spanned by them.

Due to the attention mechanism, the aggregation is invariant to the image order.
Our NAN is trained with a standard classification or verification loss without any
extra supervision signal, and we found that it automatically learns to advocate
high‐quality face images while repelling low‐quality ones such as blurred,
occluded and improperly exposed faces.

It help to conducting a certain type of pooling to aggregate the frame‐level features
together to form a video‐level representation

Such a representation would allow direct, constant‐time computation of the
similarity or distance without the need for frame‐to‐frame matching.


# Architecture:-
![nan_network](https://user-images.githubusercontent.com/26411313/56787621-bf557d00-681a-11e9-982e-4cd44de7c3b8.png)




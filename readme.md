This is the implementation of the paper " CONVOLUTIONAL NEURAL NETWORKS FOR SENTENCE CLASSIFICATION " by Y.Kim which was published in EMNLP 2014.

We use three different types of model variants and compare their performance in two different datasets : SUBJECTIVITY and SENTIMENT POLARITY on MOVIES

The models have been named as CNN_RANDOM_ , CNN_STATIC_ , CNN_MULTICHANNEL_    based on their initializations and difference in architecture.
Here we use pretrained wordvec originally given by Mikolov et. al. However there is much scope to experiment with other pretrained wordvec like Glove Vectors given by Socher et al.

In the SUBJECTIVITY dataset:
CNN_RANDOM has an accuracy of 89.7 %
CNN_STATIC has an accuracy of 91.6 %
CNN_MULTICHANNEL has an accuracy of 91.9 %. 
 
The models have been saved in the following files :
CNN_RANDOM_SUBJECTIVITY.h5  ,   CNN_STATIC_SUBJECTIVITY.h5   , CNN_MULTICHANNEL_SUBJECTIVITY.h5

We have used Early Stopping set at a patience of 8 and have used the Adadelta update rule with a learning rate of 1.0.

The code has been entirely reproduced in Keras Deep Learning library .



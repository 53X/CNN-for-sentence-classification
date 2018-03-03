#library dependencies

from keras.models import Model
from keras.layers import Input,Conv2D,Embedding,Reshape,merge,GlobalMaxPool2D,Dense,Dropout
from keras.optimizers import Adadelta
from preprocessing import Preprocessing

#global variables 

window_sizes=[3,4,5]
maxlen_of_sequence=200
feature_maps=100




#Various architectures of the model

class Models():

    def CNN_Random(vocab_size):
        filter_varying_prop=[]
        input_layer=Input(shape=(maxlen_of_sequence,),name='Input_Layer')
        embedding_layer=Embedding(vocab_size,300,input_length=maxlen_of_sequence,embeddings_initializer='uniform',trainable=True,name='Embedding_Layer')(input_layer)
        reshape_layer=Reshape(target_shape=(1,maxlen_of_sequence,300),name='Reshape_Layer')(embedding_layer)
        for size in window_sizes:
            conv_layer=Conv2D(filters=feature_maps,kernel_size=(size,300),strides=(1,1),padding='valid',data_format='channels_first',activation='relu',name='CONV_FOR_WINDOW_'+str(size))(reshape_layer)
            pooling_layer=GlobalMaxPool2D(data_format='channels_first',name='POOL_FOR_WINDOW_'+str(size))(conv_layer)
            filter_varying_prop.append(pooling_layer)
        concat_layer=merge(filter_varying_prop,mode='concat',concat_axis=-1,name='Concatenate_Layer')
        #concat_layer=Dropout(0.5)(concat_layer)
        output_layer=Dense(2,activation='softmax',name='Output_Layer')(concat_layer)
        model=Model(inputs=input_layer,outputs=output_layer)
        optimizer=Adadelta(lr=1.0)
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        return model

    def CNN_Static(vocab_size,dat='subjectivity',embed='word2vec'):
        filter_varying_prop=[]
        input_layer=Input(shape=(maxlen_of_sequence,),name='Input_Layer')
        embedding_matrix=Preprocessing.get_embedding_weights(data=dat,embedding=embed)
        embedding_layer=Embedding(vocab_size,300,input_length=maxlen_of_sequence,trainable=False,weights=[embedding_matrix],name='Embedding_Layer')(input_layer)
        reshape_layer=Reshape(target_shape=(1,maxlen_of_sequence,300),name='Reshape_Layer')(embedding_layer)
        for size in window_sizes:
            conv_layer=Conv2D(filters=feature_maps,kernel_size=(size,300),strides=(1,1),padding='valid',data_format='channels_first',activation='relu',name='CONV_FOR_WINDOW_'+str(size))(reshape_layer)
            pooling_layer=GlobalMaxPool2D(data_format='channels_first',name='POOL_FOR_WINDOW_'+str(size))(conv_layer)
            filter_varying_prop.append(pooling_layer)
        concat_layer=merge(filter_varying_prop,mode='concat',concat_axis=-1,name='Concatenate_Layer')
        #concat_layer=Dropout(0.5)(concat_layer)
        output_layer=Dense(2,activation='softmax',name='Output_Layer')(concat_layer)
        model=Model(inputs=input_layer,outputs=output_layer)
        optimizer=Adadelta(lr=1.0)
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        return model

    def CNN_MultiChannel(vocab_size,dat='subjectivity',embed='word2vec'):
        input_layer=Input(shape=(maxlen_of_sequence,),name='Input_Layer')
        embedding_weights=Preprocessing.get_embedding_weights(data=dat,embedding=embed)
        channel_finetune=Embedding(vocab_size,300,input_length=maxlen_of_sequence,weights=[embedding_weights],trainable=True,name='FineTune_Channel')(input_layer)
        channel_static=Embedding(vocab_size,300,input_length=maxlen_of_sequence,weights=[embedding_weights],trainable=False,name='Static_Channel')(input_layer)
        reshape_layer_finetune=Reshape(target_shape=(1,maxlen_of_sequence,300),name='Reshape_Layer_FineTune')(channel_finetune)
        reshape_layer_static=Reshape(target_shape=(1,maxlen_of_sequence,300),name='Reshape_Layer_Static')(channel_static)
        channels=[reshape_layer_finetune,reshape_layer_static]
        filter_varying_prop=[]
        for size in window_sizes:
            conv_value_list=[]
            for i,channel in enumerate(channels):
                if(i==0):
                    val='_FINETUNE'
                else:
                    val='_STATIC'
                conv_layer=Conv2D(filters=feature_maps,kernel_size=(size,300),strides=(1,1),padding='valid',data_format='channels_first',activation='relu',name='CONV_FOR_WINDOW_'+str(size)+val)(channel)
                conv_value_list.append(conv_layer)
            multichannel_adding_layer=merge(conv_value_list,mode='sum',name='MULTICHANNEL_ADDING_LAYER_'+str(size))
            pooling_layer=GlobalMaxPool2D(data_format='channels_first',name='POOL_FOR_WINDOW_'+str(size))(multichannel_adding_layer)
            filter_varying_prop.append(pooling_layer)
        concatenate_layer=merge(filter_varying_prop,mode='concat',concat_axis=-1,name='Concatenate_Layer')
        #concatenate_layer=Dropout(0.5)(concatenate_layer)
        output_layer=Dense(2,activation='softmax',name='Output_Layer')(concatenate_layer)
        model=Model(inputs=input_layer,outputs=output_layer)
        optimizer=Adadelta(lr=1.0)
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        return model




          


            


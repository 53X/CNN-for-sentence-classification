#library dependencies

from preprocessing import Preprocessing
from keras.callbacks import EarlyStopping,ModelCheckpoint
from labelling import Data
from architecture import Models
from keras.models import load_model
from sklearn.cross_validation import train_test_split

#train and test data 

corpus,labels,training_data,word_index=Preprocessing.create_sequences(data='sentiment')
x_train,x_test,y_train,y_test=train_test_split(training_data,labels,random_state=1999,train_size=0.8)
vocabulary=len(word_index)+1
print(vocabulary)
#getting the model variants

CNN_Random_model=Models.CNN_Random(vocabulary)
CNN_Static_model=Models.CNN_Static(vocabulary,dat='sentiment')
CNN_MultiChannel_model=Models.CNN_MultiChannel(vocabulary,dat='sentiment')

#Training the models

early_stopping=EarlyStopping(monitor='val_acc',patience=8,verbose=1,mode='auto')
model_checkpoint=ModelCheckpoint(monitor='val_acc',verbose=1,mode='auto',save_best_only=True,filepath='saved_model.h5',save_weights_only=False,period=5)
#CNN_Random_model.fit(x_train,y_train,nb_epoch=200,verbose=1,callbacks=[early_stopping,model_checkpoint],validation_split=0.2,batch_size=50)
#CNN_Static_model.fit(x_train,y_train,nb_epoch=200,verbose=1,callbacks=[early_stopping,model_checkpoint],validation_split=0.2,batch_size=5)
CNN_MultiChannel_model.fit(x_train,y_train,nb_epoch=200,verbose=1,callbacks=[early_stopping,model_checkpoint],validation_split=0.2,batch_size=5)
model=load_model('saved_model.h5')
print(model.evaluate(x_test,y_test,batch_size=5,verbose=1))




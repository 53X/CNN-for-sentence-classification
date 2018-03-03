#library dependencies

import io
import os
from keras.utils.np_utils import to_categorical


#creating the dataset

dataset=[]
home='/home/pranay'
Downloads='Downloads'
work='WorkFolder'
data='data'
subject='subjectivity_data'
sentiment='sentiment_polarity'
file_names=['subj.txt','obj.txt']
folder_names=['neg','pos']

class Data():

    def create_subjectivity_dataset():
        labels=[]
        for file in file_names:
            path=os.path.join(home,Downloads,work,data,subject,file)
            with io.open(path,'r',encoding='ascii',errors='ignore') as labelling_file:
                for lines in labelling_file:
                    dataset.append(lines)
                    if(file=='subj.txt'):
                        labels.append(0)
                    else:
                        labels.append(1)
        labels=to_categorical(labels,num_classes=2)                
        return(dataset,labels)
    
    def create_polarity_dataset():
        labels=[]
        for folder in folder_names:
            path=os.path.join(home,Downloads,work,data,sentiment,folder)
            file_list=os.listdir(path)
            for file in file_list:
                final_path=os.path.join(path,file)
                with io.open(final_path,'r',encoding='ascii',errors='ignore') as labelling_file:
                    for lines in labelling_file:
                        dataset.append(lines)
                        if(folder=='neg'):
                            labels.append(0)
                        else:
                            labels.append(1)
        labels=to_categorical(labels,num_classes=2)                    
        return(dataset,labels)                  
                        


                
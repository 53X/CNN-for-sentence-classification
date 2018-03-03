# library dependencies and imports

from labelling import Data
from embedding_dict import embedding_dict
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from numpy import random

#Preprocessing the texts for making them model ready

class Preprocessing():

	def create_sequences(data='subjectivity'):
		if(data=='subjectivity'):
			corpus,labels=Data.create_subjectivity_dataset()
		elif(data=='sentiment'):
			corpus,labels=Data.create_polarity_dataset()
		tokenizer=Tokenizer()
		tokenizer.fit_on_texts(corpus)
		word_indexing=tokenizer.word_index
		sequences=tokenizer.texts_to_sequences(corpus)
		padded_sequences=pad_sequences(sequences,maxlen=200,padding='post')
		return(corpus,labels,padded_sequences,word_indexing)

	def get_embedding_weights(data='subjectivity',embedding='word2vec'):
		if(data=='subjectivity'):
			word_index=Preprocessing.create_sequences(data='subjectivity')[-1]
		else:
			word_index=Preprocessing.create_sequences(data='sentiment')[-1]
		embedding_dictionary_model=embedding_dict.get_dict(style=embedding)
		embedding_matrix=np.zeros((len(word_index)+1,300))
		for word,i in word_index.items():
			if word in embedding_dictionary_model.vocab:
				embedding_matrix[i]=embedding_dictionary_model[word]
			else:
				embedding_matrix[i]=np.random.random(300)
		return embedding_matrix	





		



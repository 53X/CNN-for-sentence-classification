#library dependencies

from gensim.models import KeyedVectors

class embedding_dict():

    def get_dict(style='word2vec'):
        if(style=='word2vec'):
            model=KeyedVectors.load_word2vec_format('mikolov_word2vec.bin',binary=True)
        elif(style=='glove42'):
            model=KeyedVectors.load_word2vec_format('glove_42B_300d.txt',binary=False)
        elif(style=='glove840'):
            model=KeyedVectors.load_word2vec_format('glove_840B_300d.txt',binary=False)
        return(model)

            

            
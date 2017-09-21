import numpy as np
#from pandas import HDFStore,DataFrame
import pandas as pd
import json
import os
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

'''
wordnet_lemmatizer = WordNetLemmatizer()

'sieben Tage die Woche'
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')
print(snowball_stemmer.stem('presumably'))
print(wordnet_lemmatizer.lemmatize('presumably'))



print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run",'v'))
print('-----------')
print(snowball_stemmer.stem("cats"))
print(snowball_stemmer.stem("cacti"))
print(snowball_stemmer.stem("geese"))
print(snowball_stemmer.stem("rocks"))
print(snowball_stemmer.stem("python"))


print(snowball_stemmer.stem("run"))
'''

from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


#import shutil
def create_pandas_file(f_path, target_dir,lang = 'english'):
    a,b = os.path.split(f_path)
    fname = b.split('.')[0]
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    loc_file = os.path.join(target_dir, fname+'.h5')

    hdf = HDFStore(loc_file)

    stop = set(stopwords.words(lang))

    start = False
    with open(f_path, 'r') as f:
        text = f.read()

        for i, row in enumerate(text.splitlines()):
            print('line',i)

            rating = json.loads(row)['ratings']['overall']
            review = json.loads(row)['text']
            tokens = word_tokenize(review.lower())
            taggs = nltk.pos_tag(tokens)
            words = []
            for w, token_pos in enumerate(taggs):
                token, pos = token_pos
                if token not in stop:

                    pos = get_wordnet_pos(pos)
                    if pos:
                        new_token = lemmatizer.lemmatize(token, pos=pos)
                    else:
                        new_token = lemmatizer.lemmatize(token)
                    #stem_token = snowball_stemmer.stem(token)
                    print(token, new_token)

                    data_row = {'position': w, 'review_id': i,'word':new_token,'label':rating,'pos':pos}

                    words.append(data_row)
            df = pd.DataFrame(words)
            if not start:
                hdf.put('d1', df, format='table', data_columns=True)
                start = True
            else:
                hdf.append('d1',df, format='table', data_columns=True)
        if i>2:
            hdf.close()
            return


import spacy
import sqlite3

# python -m spacy.en.download all
def read_data_save_db(f_path, target_dir,lang = 'en'):
    a,b = os.path.split(f_path)
    fname = b.split('.')[0]
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    nlp = spacy.load(lang)

    loc_file = os.path.join(target_dir, fname+'.sqlite')
    if os.path.isfile(loc_file):
        os.remove(loc_file)
    con = sqlite3.connect(loc_file)


    c = 0
    with open(f_path, 'r') as f:
        text = f.read()

        for i, row in enumerate(text.splitlines()):
            print('line',i)
            #hotel
            #rating = int(json.loads(row)['ratings']['overall'])
            #review = json.loads(row)['text']
            #amazon
            rating = int(json.loads(row)['review_rating'])
            review = json.loads(row)['review_text']


            doc = nlp(review.lower())
            words = []
            for w, token_ in enumerate(doc):

                if not token_.is_stop and token_.pos_!='PUNCT':
                    '''
                    pos = get_wordnet_pos(token_.pos_)
                    if pos:
                        nltk_token = lemmatizer.lemmatize(token_.text, pos=pos)
                    else:
                        nltk_token = lemmatizer.lemmatize(token_.text)
                    
                    data_row = {'position': w, 'review_id': i,'word_':token_.text,'word_lemma':token_.lemma_,'word_nltk':nltk_token,'label':rating,'pos':token_.pos_,'tag':token_.tag_}
                    
                    if data_row['word_nltk']!=data_row['word_lemma']:
                        print(data_row)
                    '''
                    data_row = {'index' : c,'position': w, 'review_id': i, 'word_': token_.text, 'word_lemma': token_.lemma_, 'label': rating, 'pos': token_.pos_, 'tag': token_.tag_}
                    c+=1
                    words.append(data_row)
            df = pd.DataFrame(words,columns=data_row.keys())
            df.set_index('index')
            df.to_sql("worddb", con, if_exists="append")
    con.close()



def analyze_data(fname):
    con = sqlite3.connect(fname)
    df = pd.read_sql_query("SELECT * from worddb", con)

    print(df)

#read_data_save_db('/home/scstech/WORK/ovation_proj/Ovation/train.txt','T')

analyze_data('/home/scstech/WORK/ovation_proj/Ovation/utils/T/train.sqlite')







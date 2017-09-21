import spacy
import multiprocessing
import json
import pandas as pd
import os
import sqlite3

def compute(path,target_dir,n_p = 5, lang='en'):
    global i
    def spacy_reader(dataset):
        indx, row = zip(*dataset)
        rating = int(json.loads(row)['ratings']['overall'])
        review = json.loads(row)['text']

        doc = nlp(review.lower())
        #words = []
        for w, token_ in enumerate(doc):

            if not token_.is_stop and token_.pos_ != 'PUNCT':
                data_row = {'position': w, 'review_id': id, 'word_': token_.text,
                            'word_lemma': token_.lemma_,
                            'label': rating, 'pos': token_.pos_, 'tag': token_.tag_}
                return data_row


    def next_chunk(text,con, dim = 1000):
        chunk = text[i, i + dim]
        index = range[i, i + dim]
        with multiprocessing.Pool(processes=n_p) as pool:
            words = pool.map(spacy_reader,zip(index, chunk) )
            df = pd.DataFrame(words, columns=words[0].keys())
            df.to_sql("worddb", con, if_exists="append")
            i += dim
        return i<len(text)

    i = 0
    with open(path, 'r') as f:
        all_dataset = f.read().splitlines()

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    nlp = spacy.load(lang)


    a, b = os.path.split(path)
    fname = b.split('.')[0]
    loc_file = os.path.join(target_dir, fname + '.sqlite')
    print(loc_file)
    if os.path.isfile(loc_file):
        os.remove(loc_file)

    con = sqlite3.connect(loc_file)
    dim=2000
    while next_chunk(all_dataset,con):
        print(i, i+dim)


compute('/home/scstech/WORK/ovation_proj/Ovation/train.txt','hotel2')
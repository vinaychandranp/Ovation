import spacy
import multiprocessing
import json
import pandas as pd
import os
import sqlite3

nlp = spacy.load('en')

def spacy_reader(line):
    indx, row = line
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



def compute(path,target_dir,n_p = 5):
    i = 0


    def next_chunk(text,con, dim = 1000):
        nonlocal i
        chunk = text[i: i + dim]
        index = range(i, i + dim)
        data = [(a,b)for a,b in zip(index, chunk)]
        with multiprocessing.Pool(processes=n_p) as pool:
            words = pool.map(spacy_reader,data)
            df = pd.DataFrame(words, columns=words[0].keys())
            df.to_sql("worddb", con, if_exists="append")
            i += dim
        return i<len(text)


    with open(path, 'r') as f:
        all_dataset = f.read().splitlines()

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    a, b = os.path.split(path)
    fname = b.split('.')[0]
    loc_file = os.path.join(target_dir, fname + '.sqlite')
    print(loc_file)
    if os.path.isfile(loc_file):
        os.remove(loc_file)

    con = sqlite3.connect(loc_file)
    dim=50
    while next_chunk(all_dataset,con):
        print(i, i+dim)


compute('/scratch/OSA/data/datasets/hotel_reviews/train/train.txt','hotel_statistics_2')


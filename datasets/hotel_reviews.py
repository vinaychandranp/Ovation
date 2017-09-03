import os
import json
import datasets
import collections


class HotelReviews(object):
    def __init__(self, train_validation_split=None, test_split=None,
                 use_defaults=True):
        if train_validation_split is not None or test_split is not None or \
                use_defaults is False:
            raise NotImplementedError('This Dataset does not implement '
                  'train_validation_split, test_split or use_defaults as the '
                  'dataset is big enough and uses dedicated splits from '
                  'the original datasets')
        self.dataset_name = 'CMU Hotel Reviews'
        self.dataset_description = 'This dataset is from CMU. Here is the ' \
               'link to the dataset http://www.cs.cmu.edu/~jiweil/html/' \
               'hotel-review.html \nIt has 553494 Training Instances ' \
               '263568 Test Instances and 61499 Validation Instances'
        self.test_split = 'large'
        self.dataset = "hotel_reviews"
        self.dataset_path = os.path.join(datasets.data_root_directory,
                                         self.dataset)
        self.train_path = os.path.join(self.dataset_path, 'train', 'train.txt')
        self.validation_path = os.path.join(self.dataset_path, 'validation',
                                            'validation.txt')
        self.test_path = os.path.join(self.dataset_path, 'test', 'test.txt')
        self.vocab_path = os.path.join(self.dataset_path, 'vocab.txt')
        self.metadata_path = os.path.abspath(os.path.join(self.dataset_path,
                                               'metadata.txt'))
        self.w2v_path = os.path.join(self.dataset_path, 'w2v.npy')

        self.w2i, self.i2w = datasets.load_vocabulary(self.vocab_path)
        self.w2v = datasets.load_w2v(self.w2v_path)

        self.vocab_size = len(self.w2i)
        self.train = DataSet(self.train_path, (self.w2i, self.i2w))
        self.validation = DataSet(self.validation_path, (self.w2i, self.i2w))
        self.test = DataSet(self.test_path, (self.w2i, self.i2w))
        self.__refresh(load_w2v=False)

    def create_vocabulary(self, min_frequency=5, tokenizer='spacy',
                          downcase=False, max_vocab_size=None,
                          name='new', load_w2v=True):
        def line_processor(line):
            json_obj = json.loads(line)
            line = json_obj["title"] + " " + json_obj["text"]
            return line

        self.vocab_path, self.w2v_path, self.metadata_path = \
            datasets.new_vocabulary([self.train_path], self.dataset_path,
                                    min_frequency, tokenizer=tokenizer,
                                    downcase=downcase,
                                    max_vocab_size=max_vocab_size, name=name,
                                    line_processor=line_processor)
        self.__refresh(load_w2v)

    def __refresh(self, load_w2v):
        self.w2i, self.i2w = datasets.load_vocabulary(self.vocab_path)
        self.vocab_size = len(self.w2i)
        if load_w2v:
            self.w2v = datasets.preload_w2v(self.w2i)
            datasets.save_w2v(self.w2v_path, self.w2v)
        self.train.set_vocab((self.w2i, self.i2w))
        self.validation.set_vocab((self.w2i, self.i2w))
        self.test.set_vocab((self.w2i, self.i2w))


class DataSet(object):
    def __init__(self, path, vocab):

        self.path = path
        self._epochs_completed = 0
        self.vocab_w2i = vocab[0]
        self.vocab_i2w = vocab[1]
        self.datafile = None

        self.Batch = collections.namedtuple('Batch', ['text',
                                                      'sentences',
                                                      'ratings_services',
                                                      'ratings_cleanliness',
                                                      'ratings_overall',
                                                      'ratings_value',
                                                      'ratings_sleep_quality',
                                                      'ratings_rooms',
                                                      'titles',
                                                      'helpful_votes'])

    def open(self):
        self.datafile = open(self.path, 'r')

    def close(self):
        self.datafile.close()

    def remove_entities(self, data):
        entities = ['PERSON', 'NORP', 'FACILITY' , 'ORG' , 'GPE' , 'LOC' +
                    'PRODUCT' , 'EVENT' , 'WORK_OF_ART' , 'LANGUAGE' ,
                    'DATE' , 'TIME' , 'PERCENT' , 'MONEY' , 'QUANTITY' ,
                    'ORDINAL' , 'CARDINAL' , 'BOE', 'EOE']
        data_ = []
        for d in data:
            d_ = []
            for token in d:
                if token not in entities:
                    d_.append(token)
            data_.append(d_)
        return data_

    def next_batch(self, batch_size=64, seq_begin=False, seq_end=False,
                   rescale=None, pad=0, raw=False, mark_entities=False,
                   tokenizer='spacy', sentence_pad=0):
        if not self.datafile:
            raise Exception('The dataset needs to be open before being used. '
                            'Please call dataset.open() before calling '
                            'dataset.next_batch()')
        text, sentences, ratings_service, ratings_cleanliness, \
        ratings_overall, ratings_value, ratings_sleep_quality, ratings_rooms, \
        titles, helpful_votes = [], [], [], [], [], [], [], [], [], []

        while len(text) < batch_size:
            row = self.datafile.readline()
            if row == '':
                self._epochs_completed += 1
                self.datafile.seek(0)
                continue
            json_obj = json.loads(row.strip())
            text.append(datasets.tokenize(json_obj["text"], tokenizer))
            sentences.append(datasets.sentence_tokenizer((json_obj["text"])))
            ratings_service.append(datasets.rescale(
                                   json_obj["ratings"]["service"], rescale))
            ratings_cleanliness.append(datasets.rescale(
                                   json_obj["ratings"]["cleanliness"], rescale))
            ratings_overall.append(datasets.rescale(
                                   json_obj["ratings"]["overall"], rescale))
            ratings_value.append(datasets.rescale(
                                   json_obj["ratings"]["value"], rescale))
            ratings_sleep_quality.append(datasets.rescale(
                               json_obj["ratings"]["sleep_quality"], rescale))
            ratings_rooms.append(datasets.rescale(
                                   json_obj["ratings"]["rooms"], rescale))
            helpful_votes.append(json_obj["num_helpful_votes"])
            titles.append(datasets.tokenize(json_obj["title"]))

        if mark_entities:
            text = datasets.mark_entities(text)
            titles = datasets.mark_entities(titles)
            sentences = [datasets.mark_entities(sentence)
                         for sentence in sentences]

        if not raw:
            text = datasets.seq2id(text[:batch_size], self.vocab_w2i, seq_begin,
                                  seq_end)
            titles = datasets.seq2id(titles[:batch_size], self.vocab_w2i,
                                     seq_begin, seq_end)
            sentences = [datasets.seq2id(sentence, self.vocab_w2i,
                     seq_begin, seq_end) for sentence in sentences[:batch_size]]
        else:
            text = datasets.append_seq_markers(text[:batch_size],
                                               seq_begin, seq_end)
            titles = datasets.append_seq_markers(titles[:batch_size],
                                                 seq_begin, seq_end)
            sentences = [datasets.append_seq_markers(sentence, seq_begin,
                         seq_end) for sentence in sentences[:batch_size]]

        if pad != 0:
            text = datasets.padseq(text[:batch_size], pad, raw)
            titles = datasets.padseq(titles[:batch_size], pad, raw)
            sentences = [datasets.padseq(sentence, pad, raw) for sentence in
                         sentences[:batch_size]]

        batch = self.Batch(
            s1=s1s,
            s2=s2s,
            sim=datasets.rescale(sims[:batch_size], rescale, (0.0, 1.0)))
        return batch

    def set_vocab(self, vocab):
        self.vocab_w2i = vocab[0]
        self.vocab_i2w = vocab[1]

    @property
    def epochs_completed(self):
        return self._epochs_completed
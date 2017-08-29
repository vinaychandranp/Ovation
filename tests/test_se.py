import os
from utils import StackExchange
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_is_instance

class TestStackExchange(object):
    @classmethod
    def setup_class(self):
        self.se = StackExchange()
        self.se.train.open()
        self.se.validation.open()
        self.se.test.open()

    @classmethod
    def teardown_class(self):
        self.se.train.close()
        self.se.validation.close()
        self.se.test.close()

    def setUp(self):
        self.se = StackExchange()
        self.se.train.open()
        self.se.validation.open()
        self.se.test.open()

    def teardown(self):
        if 'test' in self.se.vocab_path:
            os.remove(self.se.vocab_path)
        if 'test' in self.se.metadata_path:
            os.remove(self.se.metadata_path)
        if 'test' in self.se.w2v_path:
            os.remove(self.se.w2v_path)

    def create_vocab(self, min_frequency, tokenizer, downcase,
                     max_vocab_size, name):
        self.se.create_vocabulary(min_frequency=min_frequency,
                                   tokenizer=tokenizer, downcase=downcase,
                                   max_vocab_size=max_vocab_size, name=name)

        new_vocab_file = '{}_{}_{}_{}_{}_vocab.txt'.format(
                name.replace(' ', '_'), min_frequency, tokenizer, downcase,
                max_vocab_size)
        new_w2v_file = '{}_{}_{}_{}_{}_w2v.npy'.format(
                name.replace(' ', '_'), min_frequency, tokenizer, downcase,
                max_vocab_size)
        new_metadata_file = '{}_{}_{}_{}_{}_metadata.txt'.format(
                name.replace(' ', '_'), min_frequency, tokenizer, downcase,
                max_vocab_size)

        in_new_vocab = new_vocab_file in self.se.vocab_path
        in_new_w2v = new_w2v_file in self.se.w2v_path
        in_new_metadata = new_metadata_file in self.se.metadata_path

        return new_vocab_file, new_w2v_file, new_metadata_file, in_new_vocab,\
               in_new_w2v, in_new_metadata


    def test_init(self):
        self.se = StackExchange()
        assert_not_equal(self.se, None)
        assert_equal(self.se.dataset_name, 'Semantic Text Similarity - All')
        assert_equal(self.se.test_split, 'large')
        assert_equal(self.se.vocab_size, 62451)
        assert_equal(self.se.w2v.shape[0], 62451)
        assert_equal(self.se.w2v.shape[1], 300)
        assert_equal(self.se.w2v.shape[0], len(self.se.w2i))
        assert_equal(len(self.se.w2i), len(self.se.i2w))

    def test_create_vocabulary(self):
        name, min_frequency, tokenizer, downcase, max_vocab_size = \
            'test', 10, 'spacy', True, None

        new_vocab_file, new_w2v_file, new_metadata_file, in_new_vocab, \
            in_new_w2v, in_new_metadata = self.create_vocab(min_frequency,
                                tokenizer, downcase, max_vocab_size, name)

        self.validate_vocabulary(in_new_vocab, in_new_w2v, in_new_metadata)

    def test_create_vocab_max_Vocab_size(self):
        name, min_frequency, tokenizer, downcase, max_vocab_size = \
            'test', 10, 'nltk', True, 20

        new_vocab_file, new_w2v_file, new_metadata_file, in_new_vocab, \
        in_new_w2v, in_new_metadata = self.create_vocab(min_frequency,
                                                        tokenizer, downcase,
                                                        max_vocab_size, name)

        self.validate_vocabulary(in_new_vocab, in_new_w2v, in_new_metadata)
        assert_equal(self.se.vocab_size, 24)

    def test_create_vocab_nltk_tokenizer(self):
        name, min_frequency, tokenizer, downcase, max_vocab_size = \
            'test', 10, 'nltk', True, None

        new_vocab_file, new_w2v_file, new_metadata_file, in_new_vocab, \
        in_new_w2v, in_new_metadata = self.create_vocab(min_frequency,
                                                        tokenizer, downcase,
                                                        max_vocab_size, name)
        self.validate_vocabulary(in_new_vocab, in_new_w2v, in_new_metadata)

    def test_create_vocab_default_tokenizer(self):
        name, min_frequency, tokenizer, downcase, max_vocab_size = \
            'test', 10, 'default', True, None

        new_vocab_file, new_w2v_file, new_metadata_file, in_new_vocab, \
        in_new_w2v, in_new_metadata = self.create_vocab(min_frequency,
                                                        tokenizer, downcase,
                                                        max_vocab_size, name)
        self.validate_vocabulary(in_new_vocab, in_new_w2v, in_new_metadata)

    def test_batch_size(self):
        train_batch = self.se.train.next_batch()
        validation_batch = self.se.validation.next_batch()
        test_batch = self.se.test.next_batch()
        assert_equal(len(train_batch.s1), 64)
        assert_equal(len(train_batch.s2), 64)
        assert_equal(len(train_batch.sim), 64)
        assert_equal(len(validation_batch.s1), 64)
        assert_equal(len(validation_batch.s2), 64)
        assert_equal(len(validation_batch.sim), 64)
        assert_equal(len(test_batch.s1), 64)
        assert_equal(len(test_batch.s2), 64)
        assert_equal(len(test_batch.sim), 64)

        train_batch = self.se.train.next_batch(100)
        validation_batch = self.se.validation.next_batch(100)
        test_batch = self.se.test.next_batch(100)
        assert_equal(len(train_batch.s1), 100)
        assert_equal(len(train_batch.s2), 100)
        assert_equal(len(train_batch.sim), 100)
        assert_equal(len(validation_batch.s1), 100)
        assert_equal(len(validation_batch.s2), 100)
        assert_equal(len(validation_batch.sim), 100)
        assert_equal(len(test_batch.s1), 100)
        assert_equal(len(test_batch.s2), 100)
        assert_equal(len(test_batch.sim), 100)

    def test_batch_seq_begin(self):
        train_batch = self.se.train.next_batch(seq_begin=True)
        validation_batch = self.se.validation.next_batch(seq_begin=True)
        test_batch = self.se.test.next_batch(seq_begin=True)
        assert_equal(train_batch.s1[0][0], self.se.w2i['SEQ_BEGIN'])
        assert_equal(train_batch.s2[0][0], self.se.w2i['SEQ_BEGIN'])

        assert_equal(validation_batch.s1[0][0], self.se.w2i['SEQ_BEGIN'])
        assert_equal(validation_batch.s2[0][0], self.se.w2i['SEQ_BEGIN'])

        assert_equal(test_batch.s1[0][0], self.se.w2i['SEQ_BEGIN'])
        assert_equal(test_batch.s2[0][0], self.se.w2i['SEQ_BEGIN'])


        train_batch = self.se.train.next_batch(seq_begin=True, raw=True)
        validation_batch = self.se.validation.next_batch(seq_begin=True,
                                                          raw=True)
        test_batch = self.se.test.next_batch(seq_begin=True, raw=True)
        assert_equal(train_batch.s1[0][0], 'SEQ_BEGIN')
        assert_equal(train_batch.s2[0][0], 'SEQ_BEGIN')

        assert_equal(validation_batch.s1[0][0], 'SEQ_BEGIN')
        assert_equal(validation_batch.s2[0][0], 'SEQ_BEGIN')

        assert_equal(test_batch.s1[0][0], 'SEQ_BEGIN')
        assert_equal(test_batch.s2[0][0], 'SEQ_BEGIN')


    def test_batch_pad(self):
        train_batch = self.se.train.next_batch(pad=35)
        validation_batch = self.se.validation.next_batch(pad=35)
        test_batch = self.se.test.next_batch(pad=35)
        is_valid_train_batch = self.validate_batch_pad_length(train_batch,
                                                              pad_len=35)
        is_valid_validation_batch = self.validate_batch_pad_length(
                validation_batch, pad_len=35)
        is_valid_test_batch = self.validate_batch_pad_length(test_batch,
                                                             pad_len=35)
        assert_equal(is_valid_train_batch, True)
        assert_equal(is_valid_validation_batch, True)
        assert_equal(is_valid_test_batch, True)

    def test_batch_raw(self):
        train_batch = self.se.train.next_batch(raw=True)
        validation_batch = self.se.validation.next_batch(raw=True)
        test_batch = self.se.test.next_batch(raw=True)

        assert_is_instance(train_batch.s1[0][0], str)
        assert_is_instance(train_batch.s2[0][0], str)
        assert_is_instance(validation_batch.s1[0][0], str)
        assert_is_instance(validation_batch.s2[0][0], str)
        assert_is_instance(test_batch.s1[0][0], str)
        assert_is_instance(test_batch.s2[0][0], str)

    def test_batch_rescale(self):
        train_batch = self.se.train.next_batch(rescale=(5, 10))
        validation_batch = self.se.validation.next_batch(rescale=(5, 10))
        test_batch = self.se.test.next_batch(rescale=(5, 10))

        is_valid_train_sim_range = self.validate_sim_range(train_batch,
                                                            (5, 10))
        is_valid_validation_sim_range = self.validate_sim_range(
                validation_batch, (5, 10))
        is_valid_test_sim_range = self.validate_sim_range(test_batch,
                                                            (5, 10))
        assert_equal(is_valid_train_sim_range, True)
        assert_equal(is_valid_validation_sim_range, True)
        assert_equal(is_valid_test_sim_range, True)

    def test_create_vocab_default_tokenizer(self):
        name, min_frequency, tokenizer, downcase, max_vocab_size = \
            'test', 10, 'default', True, None

        new_vocab_file, new_w2v_file, new_metadata_file, in_new_vocab, \
        in_new_w2v, in_new_metadata = self.create_vocab(min_frequency,
                                                        tokenizer, downcase,
                                                        max_vocab_size, name)
        self.validate_vocabulary(in_new_vocab, in_new_w2v, in_new_metadata)



    def validate_vocabulary(self, in_new_vocab, in_new_w2v, in_new_metadata):
        assert_equal(self.se.w2v.shape[0], len(self.se.w2i))
        assert_equal(len(self.se.w2i), len(self.se.i2w))
        assert os.path.exists(self.se.vocab_path) == True
        assert in_new_vocab == True
        assert os.path.exists(self.se.w2v_path) == True
        assert in_new_w2v == True
        assert os.path.exists(self.se.metadata_path) == True
        assert in_new_metadata == True

    def validate_batch_pad_length(self, batch, pad_len):
        valid_batch = False
        prev_instance = None
        for s1, s2 in zip(batch.s1, batch.s2):
            if prev_instance is None:
                prev_instance = (s1, s2)
                continue
            if len(prev_instance[0]) != len(s1) or len(prev_instance[1]) != \
                    len(s2) or len(s1) != pad_len or len(s2) != pad_len:
                valid_batch = False
                break
            else:
                valid_batch = True
                prev_instance = (s1, s2)
        return valid_batch

    def validate_sim_range(self, batch, val_range):
        valid_batch = False
        for sim in batch.sim:
            if sim >= val_range[0] and sim <= val_range[1]:
                valid_batch = True
            else:
                valid_batch = False
                break
        return valid_batch

# Copied from
# https://github.com/google/seq2seq/blob/master/bin/tools/generate_vocab.py
#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import argparse
import collections
import logging
import progressbar
import datasets


parser = argparse.ArgumentParser(
    description="Generate vocabulary for a tokenized text file.")
parser.add_argument(
    "--min_frequency",
    dest="min_frequency",
    type=int,
    default=0,
    help="Minimum frequency of a word to be included in the vocabulary.")
parser.add_argument(
    "--max_vocab_size",
    dest="max_vocab_size",
    type=int,
    help="Maximum number of tokens in the vocabulary")
parser.add_argument(
    "--downcase",
    dest="downcase",
    type=bool,
    help="If set to true, downcase all text before processing.",
    default=False)
parser.add_argument(
    "--semhash",
    dest="semhash",
    type=bool,
    help="If set to true, then semantic hash is generated and vocab of "
         "trigrams is generated",
    default=False)
parser.add_argument(
    "infile",
    nargs="?",
    type=argparse.FileType("r"),
    default=sys.stdin,
    help="Input tokenized text file to be processed.")
parser.add_argument(
    "--lang",
    type=str,
    default='english',
    help="Language of the tokenizer to be used (english, german, italian, ...)")
parser.add_argument(
    "--line_processor",
    type=str,
    default='sts',
    help="Function that processes each line before it is tokenized")
parser.add_argument(
    "--tokenizer",
    type=str,
    default='spacy',
    help="Tokenizer to be used (spacy or nltk)")
parser.add_argument(
    "--delimiter",
    dest="delimiter",
    type=str,
    default=" ",
    help="Delimiter character for tokenizing. Use \" \" and \"\" for word and char level respectively."
)

args = parser.parse_args()

def ISO2language_name(string):
  languages = {
    'english': 'en',
    'german' : 'de',
    'italian': 'it'
  }
  return languages.get(string)

def load_spacy_tokenizer(args):
  import spacy
  nlp = spacy.load(ISO2language_name(args.lang))
  def spacy_tokenizer_no_doc(line):
    doc = nlp.tokenizer(line)
    return [i.text for i in list(doc)]
  return spacy_tokenizer_no_doc

def load_nltk_tokenizer(args):
  from nltk.tokenize import word_tokenize
  def nltk_tokenizer_with_language(line):
    return word_tokenize(line, language=args.lang)
  return nltk_tokenizer_with_language

tokenizers = {
  'spacy': load_spacy_tokenizer,
  'nltk' : load_nltk_tokenizer
}

def get_tokenizer(args):
  load_tokenizer = tokenizers.get(args.tokenizer, None)
  if (load_tokenizer is None):
    print("ERROR: Tokenizer doesn't exist")
    sys.exit()
  return load_tokenizer(args)

tokenizer = get_tokenizer(args)

# Counter for all tokens in the vocabulary
cnt = collections.Counter()


def tokenize(seq):
  seq_tokens = []
  doc = tokenizer(seq)
  for token in doc:
#    if token.ent_type_ == '':
#      seq_tokens.append(token.text)
#    else:
#      seq_tokens.append(token.text)
    seq_tokens.append(token)
  return seq_tokens

def json_line_processor(line):
  json_obj = json.loads(line)
  line = json_obj["review_header"] + " " + json_obj["review_text"]
  return line

def sts_line_processor(line):
  return " ".join(line.split('\t')[:2])

def semhash_line_processor(line):
  return line.split('\t')[:2]


line_processors = {
  'json': json_line_processor,
  'sts' : sts_line_processor,
  'semhash' : semhash_line_processor,
}

def line_processor(line):
  processor = line_processors.get(args.line_processor, None)
  if processor is None:
    print("ERROR: line processor doesn't exist")
    sys.exit()
  return processor(line)

bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength,
                              redirect_stdout=True)
n_line = 0
for line in args.infile:
  if args.downcase:
    line = line.lower()
  if args.delimiter == "":
    tokens = list(line.strip())
  elif args.semhash:
    line1, line2 = line_processor(line)
    #seq = datasets.split_sequence(line.strip(), mode='semhash',
    #                             tokenizer=args.tokenizer, lang=args.lang)
    seq1 = '#' + line1.strip().replace(" ", "#") + '#'
    seq2 = '#' + line2.strip().replace(" ", "#") + '#'
    hash1 = [[''.join(gram) for gram in list(datasets.find_ngrams(list(seq1),
                                                                3))]]
    hash2 = [[''.join(gram) for gram in list(datasets.find_ngrams(list(seq2),
                                                                3))]]
    hash = hash1[0] + hash2[0]
    tokens = []
    for word_hash in hash:
        tokens.append(word_hash)
    print(tokens)
  else:
    line = line_processor(line)
    tokens = tokenize(line.strip())
  tokens = [_ for _ in tokens if len(_) > 0]
  cnt.update(tokens)
  n_line += 1
  bar.update(n_line)
bar.finish()

logging.info("Found %d unique tokens in the vocabulary.", len(cnt))

# Filter tokens below the frequency threshold
if args.min_frequency > 0:
  filtered_tokens = [(w, c) for w, c in cnt.most_common()
                     if c > args.min_frequency]
  cnt = collections.Counter(dict(filtered_tokens))

logging.info("Found %d unique tokens with frequency > %d.",
             len(cnt), args.min_frequency)

# Sort tokens by 1. frequency 2. lexically to break ties
word_with_counts = cnt.most_common()
word_with_counts = sorted(
    word_with_counts, key=lambda x: (x[1], x[0]), reverse=True)

# Take only max-vocab
if args.max_vocab_size is not None:
  word_with_counts = word_with_counts[:args.max_vocab_size]

entities = ['PERSON', 'NORP', 'FACILITY' , 'ORG' , 'GPE' , 'LOC' +
                    'PRODUCT' , 'EVENT' , 'WORK_OF_ART' , 'LANGUAGE' ,
                    'DATE' , 'TIME' , 'PERCENT' , 'MONEY' , 'QUANTITY' ,
                    'ORDINAL' , 'CARDINAL' , 'BOE', 'EOE']

with open('vocab.txt', 'w') as vf, open('metadata.txt', 'w') as mf:
  mf.write('word\tfreq\n')
  mf.write('PAD\t1\n')
  mf.write('SEQ_BEGIN\t1\n')
  mf.write('SEQ_END\t1\n')
  mf.write('UNK\t1\n')

  vf.write('PAD\t1\n')
  vf.write('SEQ_BEGIN\t1\n')
  vf.write('SEQ_END\t1\n')
  vf.write('UNK\t1\n')
  #for ent in entities:
  #  vf.write("{}\t{}\n".format(ent, 1))
  #  mf.write("{}\t{}\n".format(ent, 1))

  for word, count in word_with_counts:
    vf.write("{}\t{}\n".format(word, count))
    mf.write("{}\t{}\n".format(word, count))
import datasets
import tflearn
import json

import tensorflow as tf


from flask import Flask
from flask import request
from flask import Response
from datasets import HotelReviews
from models import SentenceSentimentRegressor

# Model Parameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character "
                                            "embedding (default: 300)")
tf.flags.DEFINE_boolean("train_embeddings", True, "True if you want to train "
                                                  "the embeddings False "
                                                  "otherwise")
tf.flags.DEFINE_float("dropout", 0.5, "Dropout keep probability ("
                                              "default: 1.0)")
tf.flags.DEFINE_float("l2_reg_beta", 0.0, "L2 regularizaion lambda ("
                                            "default: 0.0)")
tf.flags.DEFINE_integer("hidden_units", 128, "Number of hidden units of the "
                                             "RNN Cell")
tf.flags.DEFINE_integer("n_filters", 500, "Number of filters ")
tf.flags.DEFINE_integer("rnn_layers", 2, "Number of layers in the RNN")
tf.flags.DEFINE_string("optimizer", 'adam', "Which Optimizer to use. "
                    "Available options are: adam, gradient_descent, adagrad, "
                    "adadelta, rmsprop")
tf.flags.DEFINE_integer("learning_rate", 0.0001, "Learning Rate")
tf.flags.DEFINE_boolean("bidirectional", True, "Flag to have Bidirectional "
                                               "LSTMs")
tf.flags.DEFINE_integer("sequence_length", 100, "maximum length of a sequence")

# Training parameters
tf.flags.DEFINE_integer("max_checkpoints", 100, "Maximum number of "
                                                "checkpoints to save.")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs"
                                           " (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set "
                                    "after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many"
                                                  " steps (default: 100)")
tf.flags.DEFINE_integer("max_dev_itr", 100, "max munber of dev iterations "
                              "to take for in-training evaluation")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft"
                                                      " device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops"
                                                       " on devices")
tf.flags.DEFINE_boolean("verbose", True, "Log Verbosity Flag")
tf.flags.DEFINE_float("gpu_fraction", 0.5, "Fraction of GPU to use")
tf.flags.DEFINE_string("data_dir", "/scratch", "path to the root of the data "
                                           "directory")
tf.flags.DEFINE_string("experiment_name",
                       "AMAZON_SENTIMENT_CNN_LSTM_REGRESSION",
                       "Name of your model")
tf.flags.DEFINE_string("mode", "train", "'train' or 'test or results'")
tf.flags.DEFINE_string("dataset", "amazon_de", "'The sentiment analysis "
                           "dataset that you want to use. Available options "
                           "are amazon_de and hotel_reviews")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def initialize_tf_graph(metadata_path, w2v):
    config = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
    sess = tf.Session(config=config)
    print("Session Started")

    with sess.as_default():
        spr_model = SentenceSentimentRegressor(FLAGS.__flags)
        spr_model.show_train_params()
        spr_model.build_model(metadata_path=metadata_path,
                                  embedding_weights=w2v)
        spr_model.create_optimizer()
        print("Siamese CNN LSTM Model built")

    print('Setting Up the Model. You can do it one at a time. In that case '
          'drill down this method')
    spr_model.easy_setup(sess)
    return sess, spr_model

ds = HotelReviews()
sess, spr_model = initialize_tf_graph(ds.metadata_path, ds.w2v)
tflearn.is_training(False, session=sess)

def get_sentiment(input_str):
    global ds
    global spr_model
    global sess
    tokenized_input = [datasets.tokenize(input_str, lang='en')]
    text = datasets.seq2id(tokenized_input, ds.w2i)
    text = datasets.padseq(text, pad=150)
    sentiment = spr_model.infer(sess, text)
    return sentiment

def process_post_request(request):
    # TODO: add sanity checks for the request before moving on
    content = request.get_json()
    text = content['input']
    print('input text: '.format(text))
    response = {}
    sentiment = get_sentiment(text)
    response['score'] = sentiment
    response['reason'] = 'some reason'
    return response

def start_server(port):
    app = Flask(__name__)
    @app.route('/generateSentiment/en', methods=['POST'])
    def sentiment():
        response = process_post_request(request)
        r = Response(response=json.dumps(response, ensure_ascii=False),
                     status=200, mimetype="application/json",
                     content_type='utf-8')
        r.headers["Content-Type"] = "text/plain; charset=utf-8"
        r.encoding = 'utf8'
        return r
    app.run(port=port)

if __name__ == '__main__':
    start_server(5000)
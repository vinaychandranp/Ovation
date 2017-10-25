import os
import datetime
import datasets
import tflearn
import json
import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pyqt_fit.nonparam_regression as smooth
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from flask import Flask
from flask import request
from flask import Response
from datasets import AmazonReviewsGerman
from datasets import HotelReviews
from datasets import id2seq
from pyqt_fit import npr_methods
from models import HeirarchicalAttentionSentimentRegressor
from flask_cors import CORS, cross_origin
from tools.create_attention_graph import plot_attention
# from PIL import Image
import base64


# Model Parameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character "
                                            "embedding (default: 300)")
tf.flags.DEFINE_integer("sentiment_size", 128, "sentiment size")
tf.flags.DEFINE_integer("num_hops", 5, "num of hops")
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
tf.flags.DEFINE_float("learning_rate", 0.0001, "Learning Rate")
tf.flags.DEFINE_boolean("bidirectional", True, "Flag to have Bidirectional "
                                               "LSTMs")
tf.flags.DEFINE_integer("sequence_length", 150, "maximum length of a sequence")

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
tf.flags.DEFINE_string("mode", "train", "'train' or 'test or results, infer_server, infer'")
tf.flags.DEFINE_string("dataset", "amazon_de", "'The sentiment analysis "
                           "dataset that you want to use. Available options "
                           "are amazon_de and hotel_reviews")
tf.flags.DEFINE_string("lang", "en", "language of choice")
tf.flags.DEFINE_string("port", "8080", "Port number")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
sess, spr_model = None, None

def initialize_tf_graph(metadata_path, w2v):
    global sess
    global spr_model
    config = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
    sess = tf.Session(config=config)
    print("Session Started")

    with sess.as_default():
        spr_model = HeirarchicalAttentionSentimentRegressor(FLAGS.__flags)
        spr_model.show_train_params()
        spr_model.build_model(metadata_path=metadata_path,
                                  embedding_weights=w2v)
        spr_model.create_optimizer()
        print("Siamese CNN LSTM Model built")

    print('Setting Up the Model. You can do it one at a time. In that case '
          'drill down this method')
    spr_model.easy_setup(sess)

    return sess, spr_model


def train(dataset, metadata_path, w2v, getout=False):
    print("Configuring Tensorflow Graph")
    with tf.Graph().as_default():

        sess, spr_model = initialize_tf_graph(metadata_path, w2v)
        if getout:
            tflearn.is_training(False, session=sess)
            return

        print('Opening the datasets')
        dataset.train.open()
        dataset.validation.open()
        dataset.test.open()

        min_validation_loss = float("inf")
        avg_val_loss = 0.0
        prev_epoch = 0
        tflearn.is_training(True, session=sess)
        while dataset.train.epochs_completed < FLAGS.num_epochs:
            train_batch = dataset.train.next_batch(batch_size=FLAGS.batch_size,
                               rescale=[0.0, 1.0], pad=spr_model.args["sequence_length"])
            pco, mse, loss, step = spr_model.train_step(sess,
                                                 train_batch.text,
                                                 train_batch.ratings, train_batch.lengths,
                                                 dataset.train.epochs_completed)

            if step % FLAGS.evaluate_every == 0:
                avg_val_loss, avg_val_pco, _ = evaluate(sess=sess,
                         dataset=dataset.validation, model=spr_model,
                         max_dev_itr=FLAGS.max_dev_itr, mode='val', step=step)

            if step % FLAGS.checkpoint_every == 0:
                min_validation_loss = maybe_save_checkpoint(sess,
                    min_validation_loss, avg_val_loss, step, spr_model)

            if dataset.train.epochs_completed != prev_epoch:
                prev_epoch = dataset.train.epochs_completed
                avg_test_loss, avg_test_pco, _ = evaluate(
                            sess=sess, dataset=dataset.test, model=spr_model,
                            max_dev_itr=0, mode='test', step=step)
                min_validation_loss = maybe_save_checkpoint(sess,
                            min_validation_loss, avg_val_loss, step, spr_model)

        dataset.train.close()
        dataset.validation.close()
        dataset.test.close()


def maybe_save_checkpoint(sess, min_validation_loss, val_loss, step, model):
    if val_loss <= min_validation_loss:
        model.saver.save(sess, model.checkpoint_prefix, global_step=step)
        tf.train.write_graph(sess.graph.as_graph_def(), model.checkpoint_prefix,
                             "graph" + str(step) + ".pb", as_text=False)
        print("Saved model {} with avg_mse={} checkpoint"
              " to {}\n".format(step, val_loss,
                                model.checkpoint_prefix))
        return val_loss
    return min_validation_loss


def evaluate(sess, dataset, model, step, max_dev_itr=100, verbose=True,
             mode='val'):
    results_dir = model.val_results_dir if mode == 'val' \
        else model.test_results_dir
    samples_path = os.path.join(results_dir,
                                '{}_samples_{}.txt'.format(mode, step))
    history_path = os.path.join(results_dir,
                                '{}_history.txt'.format(mode))

    avg_val_loss, avg_val_pco = 0.0, 0.0
    print("Running Evaluation {}:".format(mode))
    tflearn.is_training(False, session=sess)

    # This is needed to reset the local variables initialized by
    # TF for calculating streaming Pearson Correlation and MSE
    sess.run(tf.local_variables_initializer())
    all_dev_review, all_dev_score, all_dev_gt = [], [], []
    dev_itr = 0
    while (dev_itr < max_dev_itr and max_dev_itr != 0) \
            or mode in ['test', 'train']:
        val_batch = dataset.next_batch(FLAGS.batch_size, rescale=[0.0, 1.0],
                                       pad=model.args["sequence_length"])
        val_loss, val_pco, val_mse, val_ratings = \
            model.evaluate_step(sess, val_batch.text, val_batch.ratings,
                                val_batch.lengths)
        avg_val_loss += val_mse
        avg_val_pco += val_pco[0]
        all_dev_review += id2seq(val_batch.text, dataset.vocab_i2w)
        all_dev_score += val_ratings.tolist()
        all_dev_gt += val_batch.ratings
        dev_itr += 1

        if mode == 'test' and dataset.epochs_completed == 1: break
        if mode == 'train' and dataset.epochs_completed == 1: break

    result_set = (all_dev_review, all_dev_score, all_dev_gt)
    avg_loss = avg_val_loss / dev_itr
    avg_pco = avg_val_pco / dev_itr
    if verbose:
        print("{}:\t Loss: {}\tPco{}".format(mode, avg_loss, avg_pco))

    with open(samples_path, 'w') as sf, open(history_path, 'a') as hf:
        for x1, sim, gt in zip(all_dev_review, all_dev_score, all_dev_gt):
            sf.write('{}\t{}\t{}\n'.format(x1, sim, gt))
        hf.write('STEP:{}\tTIME:{}\tPCO:{}\tMSE\t{}\n'.format(
                step, datetime.datetime.now().isoformat(),
                avg_pco, avg_loss))
    tflearn.is_training(True, session=sess)
    return avg_loss, avg_pco, result_set


def infer_block(sess, dataset, model, step, max_dev_itr=100, verbose=True,
             mode='val'):
    results_dir = model.val_results_dir if mode == 'val' \
        else model.test_results_dir
    samples_path = os.path.join(results_dir,'attention',
                                '{}_samples_{}.txt'.format(mode, step))
    history_path = os.path.join(results_dir,'attention',
                                '{}_history.txt'.format(mode))


    print("Running Evaluation {}:".format(mode))
    tflearn.is_training(False, session=sess)

    # This is needed to reset the local variables initialized by
    # TF for calculating streaming Pearson Correlation and MSE
    sess.run(tf.local_variables_initializer())
    all_dev_review, all_dev_score, all_dev_gt, all_dev_attention = [], [], [], []
    dev_itr = 0

    val_batch = dataset.next_batch(FLAGS.batch_size, rescale=[0.0, 1.0],
                                   pad=model.args["sequence_length"])
    results = model.infer(sess, val_batch.text, val_batch.lengths)
    all_dev_review.append(id2seq(val_batch.text, dataset.vocab_i2w))
    all_dev_score.append(results[0])
    all_dev_attention.append(np.transpose(np.array(results[1]), axes=[1,0,2]))
    all_dev_gt.append(val_batch.ratings)
    dev_itr += 1

    result_set = (all_dev_review, all_dev_score, all_dev_attention, all_dev_gt)
    np.save(samples_path.split('.')[0]+'_review.npy', all_dev_review)
    np.save(samples_path.split('.')[0]+'_score.npy', all_dev_score)
    np.save(samples_path.split('.')[0]+'_attention.npy', all_dev_attention)
    np.save(samples_path.split('.')[0]+'_gt.npy', all_dev_gt)
    with open(samples_path, 'w') as sf:
        for txt, similarity, att_big, gt_rating in zip(all_dev_review, all_dev_score,all_dev_attention, all_dev_gt):
            for x1, sim, att, gt in zip(txt, similarity, att_big, gt_rating):

                sf.write('{}\t{}\t{}\n'.format(x1, sim,
                               [item/np.sum(item) * 100 for item in np.array(att)[:,:len(x1.split(' '))]],gt ))
    tflearn.is_training(True, session=sess)
    return

def infer(dataset, metadata_path, w2v, rescale=None):
    print('Configuring Tensorflow Graph')
    with tf.Graph().as_default():
        sess, model = initialize_tf_graph(metadata_path, w2v)
        dataset.test.open()
        infer_block(sess=sess,
                    dataset=dataset.test,
                    model=model,
                    max_dev_itr=1,
                    mode='test',
                    step=-1)
        dataset.test.close()
        print('Done writing results')
    return


def test(dataset, metadata_path, w2v, rescale=None):
    print("Configuring Tensorflow Graph")
    with tf.Graph().as_default():
        sess, siamese_model = initialize_tf_graph(metadata_path, w2v)
        dataset.test.open()
        avg_test_loss, avg_test_pco, test_result_set = evaluate(sess=sess,
                                                        dataset=dataset.test,
                                                        model=siamese_model,
                                                        max_dev_itr=0,
                                                        mode='test',
                                                        step=-1)
        print('Average Pearson Correlation: {}\nAverage MSE: {}'.format(
                avg_test_pco, avg_test_loss))
        dataset.test.close()
        _, ratings, gt = test_result_set
        if rescale is not None:
            gt = datasets.rescale(gt, new_range=rescale,
                                  original_range=[0.0, 1.0])

        figure_path = os.path.join(siamese_model.exp_dir,
                                   'test_no_regression_sim.jpg')
        plt.ylabel('Ground Truth Similarities')
        plt.xlabel('Predicted  Similarities')
        plt.scatter(ratings, gt, label="Similarity", s=0.2)
        plt.savefig(figure_path)
        print("saved similarity plot at {}".format(figure_path))


def results(dataset, metadata_path, w2v, rescale=None):
    print("Configuring Tensorflow Graph")
    with tf.Graph().as_default():
        sess, siamese_model = initialize_tf_graph(metadata_path, w2v)
        dataset.test.open()
        dataset.train.open()
        avg_test_loss, avg_test_pco, test_result_set = evaluate(sess=sess,
                                    dataset=dataset.test, model=siamese_model,
                                    step=-1, max_dev_itr=0, mode='test')
        avg_train_loss, avg_train_pco, train_result_set = evaluate(sess=sess,
                                                       dataset=dataset.train,
                                                       model=siamese_model,
                                                       max_dev_itr=0,
                                                       step=-1,
                                                       mode='train')
        dataset.test.close()
        dataset.train.close()
        print('TEST RESULTS:\nMSE: {}\t Pearson Correlation: {}\n\n'
              'TRAIN RESULTS:\nMSE: {}\t Pearson Correlation: {}'.format(
                avg_test_loss, avg_test_pco, avg_train_loss, avg_train_pco
        ))

        _, train_ratings, train_gt = train_result_set
        _, test_ratings, test_gt = test_result_set
        grid = np.r_[0:1:1000j]

        if rescale is not None:
            train_gt = datasets.rescale(train_gt, new_range=rescale,
                                        original_range=[0.0, 1.0])
            test_gt = datasets.rescale(test_gt, new_range=rescale,
                                       original_range=[0.0, 1.0])
            # grid = np.r_[rescale[0]:rescale[1]:1000j]

        figure_path = os.path.join(siamese_model.exp_dir,
                                   'results_test_sim.jpg')
        reg_fig_path = os.path.join(siamese_model.exp_dir,
                                    'results_line_fit.jpg')
        plt.title('Regression Plot for Test Set Similarities')
        plt.ylabel('Ground Truth Similarities')
        plt.xlabel('Predicted  Similarities')

        print("Performing Non Parametric Regression")
        non_param_reg = non_parametric_regression(train_ratings,
                                train_gt,method=npr_methods.SpatialAverage())

        reg_test_sim = non_param_reg(test_ratings)
        reg_pco = pearsonr(reg_test_sim, test_gt)
        reg_mse = mean_squared_error(test_gt, reg_test_sim)
        print("Post Regression Test Results:\nPCO: {}\nMSE: {}".format(reg_pco,
                                                                       reg_mse))

        plt.scatter(reg_test_sim, test_gt, label='Similarities', s=0.2)
        plt.savefig(figure_path)

        plt.clf()

        plt.title('Regression Plot for Test Set Similarities')
        plt.ylabel('Ground Truth Similarities')
        plt.xlabel('Predicted  Similarities')
        plt.scatter(test_ratings, test_gt, label='Similarities', s=0.2)
        plt.plot(grid, non_param_reg(grid), label="Local Linear Smoothing",
                 linewidth=2.0, color='r')
        plt.savefig(reg_fig_path)

        print("saved similarity plot at {}".format(figure_path))
        print("saved regression plot at {}".format(reg_fig_path))


def non_parametric_regression(xs, ys, method):
    reg = smooth.NonParamRegression(xs, ys, method=method)
    reg.fit()
    return reg


def get_sentiment(input_str):
    global ds
    global spr_model
    global sess
    tokenized_text = datasets.tokenize(input_str, lang=FLAGS.lang)
    length = [len(tokenized_text)] * 128
    tokenized_input = [tokenized_text] * 128
    text = datasets.seq2id(tokenized_input, ds.w2i)
    text = datasets.padseq(text, pad=150)
    results = spr_model.infer(sess, text, length)
    merged_tokens = ' '.join(tokenized_text)
    return results[0], results[1], length, tokenized_text, merged_tokens


def process_post_request(request):
    # TODO: add sanity checks for the request before moving on
    content = request.get_json()
    text = content['input']
    print('input text: '.format(text))
    response = {}
    sentiment, attention, length, tokenized_text, merged_tokens = get_sentiment(text)
    response['score'] = str(sentiment[0])
    response['reason'] = 'some reason'
    attention = [item/np.sum(item) for item in np.array(attention)[:,0,:length[0]]]
    response['attention'] = [[str(i)for i in item] for item in attention]
    response['parsed_text'] = merged_tokens
    response['tokens'] = tokenized_text
    attn_ids_sorted = [np.argsort(attn)[::-1] for attn in attention]
    locations = []
    hop_sampled_toks = []
    for attn_ids in attn_ids_sorted:
        num = int(len(attn_ids)*0.5)
        imp_tok_ids = attn_ids[: num]
        sampled_tokens = [[tokenized_text[id], str(id)] for id in imp_tok_ids]
        location = []
        start, end = 0, 0
        """
        for t_i, tok in enumerate(tokenized_text):
            if t_i in imp_tok_ids:
                end += len(tok) - 1
                location.append([start, end])
                end += 1
                start += len(tok)
            else:
                if t_i < len(tokenized_text) - 1:
                    start += len(tok)
                    end += len(tok)
                else:
                    start += len(tok) - 1
                    end += len(tok) - 1
        locations.append(location)
        """
        hop_sampled_toks.append(sampled_tokens)
    response['sample_tokens'] = hop_sampled_toks
    #response['locations'] = locations
    """
    try:
        plot_attention(tokenized_text, attention, length[0]).savefig('/tmp/tmp.png')
        # img = Image.open('/tmp/tmp.png')
        with open('/tmp/tmp.png','rb') as f:
            response['graph'] = base64.b64encode(f.read()).decode('utf-8')
    except:
        print("Graph not generated!")
        response['graph'] = 'Graph not generated!'
    """
    return response


def start_server(port):
    app = Flask(__name__)
    cors = CORS(app)
    app.config['CORS_HEADERS'] = 'Content-Type'
    @app.route('/generateSentiment/{}'.format(FLAGS.lang), methods=['POST'])
    @cross_origin()
    def sentiment():
        response = process_post_request(request)
        r = Response(response=json.dumps(response),
                     status=200, mimetype="application/json",
                     content_type='application/json')
        r.headers["Content-Type"] = "text/plain; charset=utf-8"
        r.encoding = 'utf8'
        return r
    app.run(host='0.0.0.0', port=port)


if __name__ == '__main__':
    ds = None
    if FLAGS.dataset == 'amazon_de':
        print('Using the Amazon Reviews DE dataset')
        ds = AmazonReviewsGerman(data_balancing=True)
    elif FLAGS.dataset == 'hotel_reviews':
        print('Using the Amazon Reviews DE dataset')
        ds = HotelReviews(data_balancing=True)
    else:
        raise NotImplementedError('Dataset {} has not been '
                                  'implemented yet'.format(FLAGS.dataset))
    if FLAGS.mode == 'train':
        train(ds, ds.metadata_path, ds.w2v)
    elif FLAGS.mode == 'test':
        test(ds, ds.metadata_path, ds.w2v)
    elif FLAGS.mode == 'results':
        results(ds, ds.metadata_path, ds.w2v)
    elif FLAGS.mode == 'infer_server':
        train(ds, ds.metadata_path, ds.w2v, getout=True)
        start_server(FLAGS.port)
    elif FLAGS.mode == 'infer':
        infer(ds, ds.metadata_path, ds.w2v)
import datetime

import tensorflow as tf

from utils import ops
from utils import distances
from utils import losses
from scipy.stats import pearsonr
from tflearn.layers.core import dropout
from sklearn.metrics import mean_squared_error
from tensorflow.contrib.tensorboard.plugins import projector

from models.model import Model

class CNNLSTMMargin(Model):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses a word embedding layer, followed by a bLSTM and a simple Energy Loss
    layer.
    """

    def create_placeholders(self):

        # A tensorflow Placeholder for the 1st input sentence. This
        # placeholder would expect data in the shape [BATCH_SIZE X
        # SEQ_MAX_LENGTH], where each row of this Tensor will contain a
        # sequence of token ids representing the sentence
        self.input_s1 = tf.placeholder(tf.int32, [None,
                                              self.args.get("sequence_length")],
                                       name="input_s1")

        # This is similar to self.input_s1, but it is used to feed the second
        #  sentence
        self.input_s2 = tf.placeholder(tf.int32, [None,
                                              self.args.get("sequence_length")],
                                       name="input_s2")

        self.input_s3 = tf.placeholder(tf.int32, [None,
                                                  self.args.get("sequence_length")],
                                       name="input_s3")

        # This is a placeholder to feed in the ground truth similarity
        # between the two sentences. It expects a Matrix of shape [BATCH_SIZE]
        self.input_sim = tf.placeholder(tf.float32, [None], name="input_sim")

    def build_model(self, metadata_path=None, embedding_weights=None,
                    vocab_size=None, embedding_size=300):
        """
        This method builds the computation graph by adding layers of
        computations. It takes the metadata_path (of the dataset vocabulary)
        and a preloaded word2vec matrix and input and uses them (if not None)
        to initialize the Tensorflow variables. The metadata is used to
        visualize the word embeddings that are being trained using Tensorflow
        Projector. Additionally you can use any other tool to visualize them.
        https://www.tensorflow.org/versions/r0.12/how_tos/embedding_viz/
        :param metadata_path: Path to the metadata of the vocabulary. Refer
        to the datasets API
        https://github.com/mindgarage/Ovation/wiki/The-Datasets-API
        :param embedding_weights: the preloaded w2v matrix that corresponds
        to the vocabulary. Refer to https://github.com/mindgarage/Ovation/wiki/The-Datasets-API#what-does-a-dataset-object-have
        :return:
        """
        # Build the Embedding layer as the first layer of the model

        self.embedding_weights, self.config = ops.embedding_layer(
                                        metadata_path, vocab_size=vocab_size,
            embedding_shape=self.args["embedding_dim"])
        self.embedded_s1 = tf.nn.embedding_lookup(self.embedding_weights,
                                                  self.input_s1)
        self.embedded_s2 = tf.nn.embedding_lookup(self.embedding_weights,
                                                      self.input_s2)

        self.embedded_s3 = tf.nn.embedding_lookup(self.embedding_weights,
                                                  self.input_s3)

        
        self.s1_cnn_out = ops.multi_filter_conv_block(self.embedded_s1,
                                self.args["n_filters"],
                                dropout_keep_prob=self.args["dropout"])
        self.s1_lstm_out = ops.lstm_block(self.s1_cnn_out,
                                   self.args["hidden_units"],
                                   dropout=self.args["dropout"],
                                   layers=self.args["rnn_layers"],
                                   dynamic=False,
                                   bidirectional=self.args["bidirectional"])

        self.s2_cnn_out = ops.multi_filter_conv_block(self.embedded_s2,
                                      self.args["n_filters"], reuse=True,
                                      dropout_keep_prob=self.args["dropout"])
        self.s2_lstm_out = ops.lstm_block(self.s2_cnn_out,
                                   self.args["hidden_units"],
                                   dropout=self.args["dropout"],
                                   layers=self.args["rnn_layers"],
                                   dynamic=False, reuse=True,
                                   bidirectional=self.args["bidirectional"])

        self.s3_cnn_out = ops.multi_filter_conv_block(self.embedded_s3,
                                                      self.args["n_filters"], reuse=True,
                                                      dropout_keep_prob=self.args["dropout"])
        self.s3_lstm_out = ops.lstm_block(self.s3_cnn_out,
                                          self.args["hidden_units"],
                                          dropout=self.args["dropout"],
                                          layers=self.args["rnn_layers"],
                                          dynamic=False, reuse=True,
                                          bidirectional=self.args["bidirectional"])

        s1_drop = dropout(self.s1_lstm_out, 0.2)
        s2_drop = dropout(self.s2_lstm_out, 0.2)
        s3_drop = dropout(self.s3_lstm_out, 0.2)

        self.good_sim = tf.squeeze(distances.gesd(s1_drop, s2_drop))
        self.bad_sim = tf.squeeze(distances.gesd(s1_drop, s3_drop))
    
        with tf.name_scope("loss"):
            self.loss = losses.margin_loss(self.good_sim, self.bad_sim,
                                           margin=self.args.get("margin", 0.05))

            if self.args["l2_reg_beta"] > 0.0:
                self.regularizer = ops.get_regularizer(self.args["l2_reg_beta"])
                self.loss = tf.reduce_mean(self.loss + self.regularizer)


    def create_scalar_summary(self, sess):
        """
        This method creates Tensorboard summaries for some scalar values
        like loss and pearson correlation
        :param sess:
        :return:
        """
        # Summaries for loss and accuracy
        self.loss_summary = tf.summary.scalar("loss", self.loss)


        # Train Summaries
        self.train_summary_op = tf.summary.merge([self.loss_summary])

        self.train_summary_writer = tf.summary.FileWriter(self.checkpoint_dir,
                                                     sess.graph)
        projector.visualize_embeddings(self.train_summary_writer,
                                       self.config)

        # Dev summaries
        self.dev_summary_op = tf.summary.merge([self.loss_summary])

        self.dev_summary_writer = tf.summary.FileWriter(self.dev_summary_dir,
                                                   sess.graph)

    def train_step(self, sess, s1_batch, s2_batch, s3_batch, sim_batch,
                   epochs_completed, verbose=True):
            """
            A single train step
            """

            # Prepare data to feed to the computation graph
            feed_dict = {
                self.input_s1: s1_batch,
                self.input_s2: s2_batch,
                self.input_s3: s3_batch,
                self.input_sim: sim_batch,
            }

            # create a list of operations that you want to run and observe
            ops = [self.tr_op_set, self.global_step, self.loss, self.good_sim, self.bad_sim]

            # Add summaries if they exist
            if hasattr(self, 'train_summary_op'):
                ops.append(self.train_summary_op)
                _, step, loss, good_sim, bad_sim, summaries = sess.run(ops,
                    feed_dict)
                self.train_summary_writer.add_summary(summaries, step)
            else:
                _, step, loss, good_sim, bad_sim = sess.run(ops, feed_dict)

            if verbose:
                time_str = datetime.datetime.now().isoformat()
                print("Epoch: {}\tTRAIN {}: Current Step{}\tLoss{:g}".format(epochs_completed,
                        time_str, step, loss))
            return loss, step, good_sim, bad_sim

    def evaluate_step(self, sess, s1_batch, s2_batch, s3_batch, sim_batch, verbose=True):
        """
        A single evaluation step
        """

        # Prepare the data to be fed to the computation graph
        feed_dict = {
            self.input_s1: s1_batch,
            self.input_s2: s2_batch,
            self.input_s3: s3_batch,
            self.input_sim: sim_batch
        }

        # create a list of operations that you want to run and observe
        ops = [self.global_step, self.loss, self.good_sim, self.bad_sim]

        # Add summaries if they exist
        if hasattr(self, 'dev_summary_op'):
            ops.append(self.dev_summary_op)
            step, loss, good_sim, bad_sim, summaries = sess.run(ops,
                                                                  feed_dict)
            self.dev_summary_writer.add_summary(summaries, step)
        else:
            step, loss, good_sim, bad_sim = sess.run(ops, feed_dict)

        time_str = datetime.datetime.now().isoformat()

        if verbose:
            print("EVAL: {}\tStep: {}\tloss: {:g}".format(
                    time_str, step, loss))
        return loss, good_sim, bad_sim


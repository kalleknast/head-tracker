# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:17:35 2016

@author: hjalmar
"""

import tensorflow as tf
import numpy as np
from tables import openFile

class HeadTracker:
    """
    """
    def __init__(self, model_dir=None, train_data_fname=None):
    
        if not train_data_fname is None:
            self._train_data_f = openFile(train_data_fname, 'r')
            self.train_data_fname = train_data_fname
            self.model_dir = '/home/hjalmar/head-tracker_data/models/head_detect'
            self.model_dir = self.model_dir.rstrip('/')
        
        self.im_sz = self._train_data_f.root.data.shape[1:]
        self.isconfigured = False
    
    def _read_minibatch(self):
        """
        """
        if len(self._batch_idx) < 1:
            self._train_data_idx = np.random.permutation(self._train_data_idx)
            self._batch_idx = list(range(0,
                                         self.train_data_N - self.batch_sz,
                                         self.batch_sz))
            self.epoch_n += 1
        b_idx = self._batch_idx.pop()
        idx = self._train_data_idx[b_idx: b_idx + self.batch_sz]        
        X = self._train_data_f.root.data[idx, :, :].reshape((-1, 32, 32, 1))
        y = self._train_data_f.root.labels[idx, 0]       
        y = (np.arange(self.num_labels) == y[:,None]).astype(np.float32)
        
        return X, y
        
    def _read_valid_data(self):
        
        X = self._train_data_f.root.data[self._valid_data_idx, :, :].reshape((-1, 32, 32, 1))
        y = self._train_data_f.root.labels[self._valid_data_idx, 0]
        y = (np.arange(self.num_labels) == y[:,None]).astype(np.float32)        
        
        return X, y

    def _read_test_data(self):
        
        X = self._train_data_f.root.data[self._test_data_idx, :, :].reshape((-1, 32, 32, 1))
        y = self._train_data_f.root.labels[self._test_data_idx, 0]
        y = (np.arange(self.num_labels) == y[:,None]).astype(np.float32)        
        
        return X, y

    def train(self, batch_size=128, num_epochs=10,
              acc_threshold=95, verbose=True):
        """
        """
        if not self._train_data_f.isopen:
            self._train_data_f = openFile(self.train_data_fname, 'r')
        
        if not self.isconfigured:
            self.configure()

        self.epoch_N = num_epochs
        self.batch_sz = batch_size        
        self.acc_threshold = acc_threshold
        self.data_N = self._train_data_f.root.data.shape[0]
        self.epoch_n = 0
        self._batch_idx = []
        keep_prob = self.keep_prob
        self.records = {'loss': [], 
                        'minibatch_acc': [],
                        'validation_acc': [],
                        'test_acc': 0.0}
        
        # Only when initializing traning.
        self._split_dataset(train_frac=0.6, valid_frac=0.2, test_frac=0.2)        
        self.num_batches_per_epoch = self.train_data_N // self.batch_sz
        self.num_batches = self.num_batches_per_epoch * self.epoch_N
        
        if verbose:
            print('%s|%s|%s|%s\n%s' % ('Epoc'.center(8),
                                       'Minibatch loss'.center(19), 
                                       'Minibatch accuracy'.center(22),
                                       'Validation accuracy'.center(23),
                                       '-'*75))
        # Training loop
        with tf.Session(graph=self.graph) as session:
            session.run(self._init_op)

            for i in range(self.num_batches):
                # run training
                X, y = self._read_minibatch()
                feed_dict = {self._X: X, self._y: y, keep_prob: 0.5}
                _, l, predictions = session.run([self.optimizer,
                                                 self.loss,
                                                 self.predictions],
                                                 feed_dict=feed_dict)
                # TODO: per epox or fraction of epoc
                if (i % self.num_batches_per_epoch == 0):
                    self.records['loss'].append(l)
                    feed_dict[keep_prob] = 1.0
                    mb_acc = self.accuracy(self.predictions.eval(feed_dict=feed_dict), y)
                    self.records['minibatch_acc'].append(mb_acc)
                    X, y = self._read_valid_data()
                    feed_dict = {self._X: X, self._y: y, keep_prob: 1.0}
                    v_acc = self.accuracy(self.predictions.eval(feed_dict=feed_dict), y)
                    self.records['validation_acc'].append(v_acc)
                    if verbose:
                        print('   %-5d|       %-12.3f|        %-14.2f|       '
                              '%-16.2f' % (self.epoch_n, l, mb_acc, v_acc))
        
            X, y = self._read_test_data()
            feed_dict = {self._X: X, self._y: y, keep_prob: 1.0}
            t_acc = self.accuracy(self.predictions.eval(feed_dict=feed_dict), y)
            self.records['test_acc'] = t_acc

            if verbose:
                print('\n  Test accuracy: %1.3f%%' % t_acc)
                
            if v_acc > self.acc_threshold:
                # Save the variables to disk.
                model_fn = ('%s/head_detect_model_accd%1.1f' % 
                            (self.model_dir, round(v_acc, 1)))
                save_path = self._saver.save(session, model_fn + '.ckpt')
                f_conf = open(model_fn + '.conf', 'w')
                s = ('batch_sz: %d\ndata_N: %d\nepoch_N: %d\n'
                    'num_labels: %d\ntest_data_N: %d\n'
                    'test_data_frac: %1.3f\nvalid_data_N: %d\n'
                    'valid_data_frac: %1.3f\ntrain_data_N: %d\n'
                    'train_data_frac: %1.3f' %
                    (self.batch_sz, self.data_N,
                     self.epoch_N, self.num_labels,
                     self.test_data_N, self.test_data_frac,
                     self.valid_data_N, self.valid_data_frac,
                     self.train_data_N, self.train_data_frac))
                f_conf.write(s)
                f_conf.close()
                print("\n Model saved in file:\n\t%s" % save_path)
                print(" Model settings saved in file:\n\t%s" %
                      (model_fn + '.conf'))

            self.close()

    def close(self):
        """
        """
        self._train_data_f.close()

    def accuracy(self, predictions, labels):
        acc = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
        return (acc * 100) / predictions.shape[0]

    def _split_dataset(self, train_frac=0.6, valid_frac=0.2, test_frac=0.2):
        """
        """
        if not (train_frac + valid_frac + test_frac) == 1.0:
            print('The fractions of training, validation and test data have '
                  'to sum to 10')
            return 0

        self.train_data_frac = train_frac
        self.valid_data_frac = valid_frac
        self.test_data_frac = test_frac

        self.train_data_N = int(round(self.train_data_frac * self.data_N))
        self.valid_data_N = int(round(self.valid_data_frac * self.data_N))
        self.test_data_N = int(round(self.test_data_frac * self.data_N))
        
        self._data_idx = np.random.permutation(range(self.data_N))
        i1 = self.train_data_N
        self._train_data_idx = self._data_idx[:i1]
        i0, i1 = i1, i1 + self.valid_data_N
        self._valid_data_idx = self._data_idx[i0: i1]
        i0 = i1
        self._test_data_idx = self._data_idx[i0:]

        
    def configure(self, num_classes=2):
        """
        Convnet with max pooling and dropout.
        """
        # TODO: Fix proper settings for num_steps
        num_steps = 3001

        patch_size = 5
        depth = 16
        num_hidden = 64
        self.num_labels = 2

        self.graph = tf.Graph()

        with self.graph.as_default():
            # Input data.
            self._X = tf.placeholder(tf.float32, shape=(None,
                                                        self.im_sz[0],
                                                        self.im_sz[1],
                                                        1))
            self._y = tf.placeholder(tf.float32, shape=(None, self.num_labels))
            # Dropout
            self.keep_prob = tf.placeholder(tf.float32)
            # Variables.
            l1_W = tf.Variable(tf.truncated_normal([patch_size,
                                                    patch_size,
                                                    1,
                                                    depth], stddev=0.07))
            l1_B = tf.Variable(tf.zeros([depth]))
            l2_W = tf.Variable(tf.truncated_normal([patch_size,
                                                    patch_size,
                                                    depth,
                                                    depth], stddev=0.02))
            l2_B = tf.Variable(tf.constant(1.0, shape=[depth]))
            l3_W = tf.Variable(tf.truncated_normal([self.im_sz[0]//4 * self.im_sz[1]//4 * depth,
                                                    num_hidden], stddev=0.006))
            l3_B = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
            l4_W = tf.Variable(tf.truncated_normal([num_hidden, self.num_labels],
                                                   stddev=0.06))
            l4_B = tf.Variable(tf.constant(1.0, shape=[self.num_labels]))
      
            # Model.
            def model(data):
                conv = tf.nn.conv2d(data, l1_W, [1, 1, 1, 1], padding='SAME')
                pool1 = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME', name='pool1')
                hidden = tf.nn.relu(pool1 + l1_B)
                conv = tf.nn.conv2d(hidden, l2_W, [1, 1, 1, 1], padding='SAME')
                pool2 = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME', name='pool2')
                hidden = tf.nn.relu(pool2 + l2_B)
                shape = hidden.get_shape().as_list()
                hidden_flat = tf.reshape(hidden, [-1, shape[1] * shape[2] * shape[3]])
                hidden = tf.nn.relu(tf.matmul(hidden_flat, l3_W) + l3_B)
                hidden_drop = tf.nn.dropout(hidden, self.keep_prob)
                return tf.matmul(hidden_drop, l4_W) + l4_B
  
            # Training computation.
            logits = model(self._X)
            self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits, self._y))
            # Optimizer.
            global_step = tf.Variable(0)  # count the number of steps taken.
            lrate = tf.train.exponential_decay(0.1, global_step, num_steps, 0.86)    
            self.optimizer = tf.train.GradientDescentOptimizer(lrate).minimize(self.loss,
                                                                               global_step=global_step)    
            # Predictions for the training, validation, and test data.
            self.predictions = tf.nn.softmax(logits)
            self._init_op = tf.initialize_all_variables()
            # Add ops to save and restore all the variables.
            self._saver = tf.train.Saver()
            self.isconfigured = True

        
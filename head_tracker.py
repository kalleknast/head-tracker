# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:17:35 2016

@author: hjalmar
"""

import tensorflow as tf
#from ht_helper import FrameStepper  # Import has to come after tensorflow bc bug/crash
from ht_helper import MultiResolutionPyramid, closest_coordinate,  FrameStepper
from data_preparation import read_log_data
import numpy as np

from scipy.misc import imresize
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as manimation
import time

class TrainHeadTrackerMRP:
    """
    """
    def __init__(self, train_fname=None,  valid_fname=None,  model_dir=None):
    
        if train_fname is None:
            train_fname = '/home/hjalmar/head_tracker/data/train_ht_overlap_level1.tfrecords'
        if valid_fname is None:
            valid_fname = '/home/hjalmar/head_tracker/data/valid_ht_overlap_level1.tfrecords'

        if model_dir is None:
            model_dir = '/media/hjalmar/VitaminD/head_tracker/model'
        self.model_dir = model_dir.rstrip('/')
        
        self.N_class = 2
        self.N_levels = 4
        self.level_done = -1
        self.level = -1
        self.levels = []
        self.current = None
        self.data = None
        self.labels = None
        win_shapes = [(48, 48),  (32, 32),  (32, 32),  (32, 32)]
                
        for k in range(self.N_levels):
            self.levels.append(levelClass(k,  win_shapes[k]))
            
    
    def get_valid_data(self,  valid_fname):
        
        images,  labels = self.get_inputs(valid_fname,  train=False)
        
        N_valid = 30000
        batch_sz = 128
        valid_images,  valid_labels = [],  []
        N_iter = int(np.ceil(N_valid / batch_sz))
        
        with tf.Session() as sess:
            
            sess.run(tf.initialize_all_variables())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,  coord=coord)   
            
            try:               
                i = 0
                while i < N_iter and not coord.should_stop():
                    imgs,  lbls = sess.run([images,  labels])            
                    valid_images.extend(imgs)
                    valid_labels.extend(lbls)
                    i += 1   
            except Exception as e: 
                coord.request_stop(e)
                    
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
            
        return np.array(valid_images),  np.array(valid_labels)
        
            
    def get_inputs(self,  fname,  batch_sz=128,  train=False, flip=True):
        """
        """
        if not tf.gfile.Exists(fname):
            raise ValueError('Failed to find file: %s' % fname)
        
        if train:
            N_epoch = self.current.N_epoch
        else:
            N_epoch = 1
            
        with tf.name_scope('input'):
            fname_queue = tf.train.string_input_producer(
                [fname], num_epochs=N_epoch)

            # Even when reading in multiple threads, share the filename
            # queue.
            image, label = self._read_and_decode(fname_queue)
            #print(image)
            if train:
                # Distort image
                image = self._distort_inputs(image,  flip=flip)
                num_examples_per_epoch = 110000  # TODO: fixe proper num training examples
                n_threads = 8
            else:
                num_examples_per_epoch = 30000  # TODO: fix this too
                n_threads = 4
                
            # Shuffle the examples and collect them into batch_sz batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            # Ensures a minimum amount of shuffling of examples.
                
            min_queue_examples = int(num_examples_per_epoch * 0.4)
            
            images, labels = tf.train.shuffle_batch( [image, label],
                                                                           batch_size=batch_sz,
                                                                           num_threads=n_threads,
                                                                           capacity=min_queue_examples + 3 * batch_sz,
                                                                           min_after_dequeue=min_queue_examples)
                                                                      
            return images, labels            


    def evaluate(self,  valid_fname,  fname_ckpt,  model_spec=None,  batch_sz=128):
        
        self.current.N_class = 2
        
          #self.session = tf.Session(graph=self.model.graph)
        # Restore variables from disk.
        # self.model.saver.restore(self.session, self.model.fname)
        
        with tf.Graph().as_default():

            images,  labels = self.get_inputs(valid_fname,  train=False)
            logits = inference(images,  self.current,  train=False, model_spec=model_spec) 
            correct = tf.equal(tf.argmax(logits, 1), tf.cast(labels,  tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            saver = tf.train.Saver()
            #saver = tf.train.Saver(tf.all_variables())
            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
                saver.restore(sess, fname_ckpt)
                
                #import pdb
                #pdb.set_trace()
                
                # Start the queue runners.
                acc = 0.0
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess,  coord=coord)               
                #print(sess.run(labels))
                try:
                    num_iter = int(np.ceil(30000 / 128))
                    step = 0
                    #print(sess.run([correct]))
                    while step < num_iter and not coord.should_stop():
                        
                        a = sess.run(accuracy)
                        print(a)
                        acc += a
                        step += 1
                        
                except Exception as e:  # pylint: disable=broad-except
                    coord.request_stop(e)

                coord.request_stop()
                coord.join(threads, stop_grace_period_secs=10)

        return 100 * (acc/step)
    
    
    def _read_and_decode(self,  fname_queue):
        reader = tf.TFRecordReader()
        _,  serialized_example = reader.read(fname_queue)
        features = tf.parse_single_example(
            serialized_example, 
            features={'image_raw': tf.FixedLenFeature([],  tf.string), 
                              'label': tf.FixedLenFeature([],  tf.int64)
                              })
  
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([self.current.win_nr *  self.current.win_nc])
        image = tf.reshape(image,  [self.current.win_nr,  self.current.win_nc,  1])
      
        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5        
        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['label'], tf.int32)
        #        y = self.labels[idx, 0]       
        #y = (np.arange(self.N_class) == y[:,None]).astype(np.float32)

        return image, label
            
    
    def _distort_inputs(self,  image,  flip=True):
        """
        """ 
        # Randomly flip the image horizontally.
        if flip:
            image = tf.image.random_flip_left_right(image)
        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        # Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_whitening(image)
        
        return image
           
        
#    def train(self, level='all', acc_threshold=98, batch_sz=128,
#              N_epoch=10, stop_threshold=True, verbose=True,  model_spec=None):
#        """
#        """
#                            
#        if level == 'all':
#
#            for level in range(self.N_levels):
#                self.level = level
#                self.current = self.levels[level]
#                self.current.N_epoch = N_epoch
#                self.current.batch_sz = batch_sz
#                self.current.acc_threshold = acc_threshold 
#                self.data = self.current.data.read()
#                self.labels = self.current.labels.read() 
#
#                self._train(verbose=verbose, stop_threshold=stop_threshold)
#        else:
#            try:
#                level = int(level)
#                self.current = self.levels[level]
#                if level != self.level:
#                    self.data = self.current.data.read()
#                    self.labels = self.current.labels.read()
#            except:
#                ValueError('Argument "level" must be either the string "all"'
#                           'or 0 to 3 as int, string or float.')
#            
#            self.current.N_positive = (self.labels == 1).sum()
#            self.current.N_negative = (self.labels == 0).sum()
#            self.level = level
#            self.current.N_epoch = N_epoch
#            self.current.batch_sz = batch_sz
#            self.current.acc_threshold = acc_threshold
#            self.current.records['loss'] = []
#            self.current.records['epoch_acc'] = []
#            self.current.records['validation_acc'] = []
#            self._train(verbose=verbose, stop_threshold=stop_threshold)


    def train(self, level,  N_epoch,  model_spec=None):
        """
        """
        model_fname = '/home/hjalmar/head_tracker/model/position_level%d' % level
        train_fname = '/home/hjalmar/head_tracker/data/train_ht_level%d.tfrecords' % level
        valid_fname = '/home/hjalmar/head_tracker/data/train_ht_level%d.tfrecords' % level
        self.current = self.levels[level]
        if level == 0:
            flip = False
        else:
            flip = True
        #threshold = self.current.acc_threshold
               
        #records = self.current.records
        self.current.batch_sz = 128
        self.current.N_epoch = N_epoch
        batch_sz = self.current.batch_sz
        #N_test, N_valid = self.current.N_test, self.current.N_valid 
        #
        N_valid = 30000
        self.current.N_train = 100000
        self.current.N_batches_per_epoch = self.current.N_train // batch_sz
        self.current.N_batches = self.current.N_batches_per_epoch * N_epoch
                
        # Get graph/model
        self.current.N_class = self.N_class
        #model = create_graph_position(self.current, model_spec=model_spec)
        learning_rate = 1e-4
                
        valid_X,  valid_y = [],  []
        
        model = create_graph_position(self.current, model_spec=model_spec)        
        
        with model.graph.as_default():
            # Input images and labels.
            images, labels = self.get_inputs(train_fname,  batch_sz=batch_sz, train=True,  flip=flip)
            valid_images, valid_labels = self.get_inputs(valid_fname,  train=False,  batch_sz=1000)        
            saver = model.saver

            with tf.Session(graph=model.graph) as session:
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.loss) 
                session.run(tf.initialize_all_variables())
                # Start input enqueue threads
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=session,  coord=coord)  
      
                validation_accuracy = []
                train_accuracy = []                
                            
                # Build the summary operation based on the TF collection of Summaries.
                #summary_op = tf.merge_all_summaries()
        

                # Instantiate a SummaryWriter to output summaries and the Graph.
                #summary_writer = tf.train.SummaryWriter('/home/hjalmar/head_tracker/model', sess.graph)                        
                #model_fname = '/home/hjalmar/head_tracker/model/ht_level%d.ckpt' % self.current.level
                
                try:
                    step, epoch_n  = 0,  0
                    epoch_loss,  epoch_acc = 0.0,  0.0
                    while (epoch_n < N_epoch) and not coord.should_stop():
                        step += 1                        
                        # Train
                        X,  y = session.run([images,  labels])      
                        feed_dict = {model.X: X, model.y_: y, model.keep_prob: 0.5}
                        optimizer.run(feed_dict)
                        epoch_loss += model.loss.eval(feed_dict=feed_dict)
                        feed_dict[model.keep_prob] = 1.0
                        epoch_acc += model.accuracy.eval(feed_dict=feed_dict)
                        
                        if (step % self.current.N_batches_per_epoch == 0):
                            epoch_n += 1
                            epoch_acc /= self.current.N_batches_per_epoch
                            print('   %-5d|       %-12.3f|        %-14.2f'  %  (epoch_n, epoch_loss, epoch_acc*100))

                            if (epoch_n % 10 == 0):
                                v_acc,  i = 0.0,  0
                                if len(valid_y) < 1:
                                    load_valid = True
                                else:
                                    load_valid = False
                                while i < N_valid // 1000:
                                    if load_valid and not coord.should_stop():
                                        X,  y = session.run([valid_images,  valid_labels]) 
                                        valid_X.append(X)
                                        valid_y.append(y)
                                    feed_dict = {model.X: valid_X[i], model.y_: valid_y[i], model.keep_prob: 1.0}
                                    v_acc += model.accuracy.eval(feed_dict=feed_dict)
                                    i += 1
                                validation_accuracy.append(100 * (v_acc/i))
                                train_accuracy.append(epoch_acc*100)                                
                                saver.save(session, '%s_acc%1.1f.ckpt' % (model_fname,  validation_accuracy[-1]), global_step=epoch_n)

                            epoch_loss,  epoch_acc = 0.0,  0.0                          
                        
                except tf.errors.OutOfRangeError:
                    print('Done training for %d epochs, %d steps.' % (epoch_n, step-1))       
                                            
                finally:
                    v_acc,  i = 0.0,  0
                    while i < N_valid // 1000:
                        if load_valid and not coord.should_stop():
                            X,  y = session.run([valid_images,  valid_labels]) 
                            valid_X.append(X)
                            valid_y.append(y)
                        feed_dict = {model.X: valid_X[i], model.y_: valid_y[i], model.keep_prob: 1.0}
                        v_acc += model.accuracy.eval(feed_dict=feed_dict)
                        i += 1
                    validation_accuracy.append(100 * (v_acc/i))
                    saver.save(session, '%s_acc%1.1f.ckpt' % (model_fname,  validation_accuracy[-1]), global_step=epoch_n)
                
                    # Ask threads to stop
                    coord.request_stop()

                # Wait for threads to finish.
                coord.join(threads)
                session.close() 
                
                print('Training accuracy:',  train_accuracy)
                print('Validation accuracy:',  validation_accuracy)
                
        return validation_accuracy[-1]
                                        
                        
    def plot(self, fname=None):
        """
        """
        
        fig = plt.figure()
        axl = fig.add_subplot(111)
        axl.plot( self.current.records['epoch_acc'], 'g', label='mean minibatch accuracy')
        axl.plot( self.current.records['validation_acc'], 'r', label='validation accuracy')
        axr = axl.twinx()
        axr.plot( self.current.records['loss'], 'k', label='loss')
        
        axl.set_title('Level %d, keep_prob: %1.2f' % (self.level, self.keep_prob))
        axl.set_xlabel('Epoch')
        axl.set_ylabel('Accuracy')
        axr.set_ylabel('Loss')
        axr.set_ylim([0, 300])
        chance_acc = 100 * (self.current.N_negative / self.current.N_data)
        axl.plot([0,  len(self.current.records['epoch_acc'])],  [chance_acc,  chance_acc],  '--r')
        axl.set_ylim([5*np.floor(chance_acc/5), 100])
        axl.grid()
        axr.legend(loc=4, fontsize=10, frameon=False)
        axl.legend(loc=3, fontsize=10, frameon=False)

        if not fname is None:
            fig.savefig(fname)
            plt.close(fig)
            
            
def train_all(valid_thresh=99):
    
    data_fname = '/home/hjalmar/head_tracker/data/ht_data_overlap_mrp.hdf'
    model_dir = '/home/hjalmar/head_tracker/model'
    fig_dir = '/home/hjalmar/head_tracker/figs'
    ht = TrainHeadTrackerMRP(model_dir=model_dir, data_fname=data_fname)
    
    #acc_thresholds = [98, 95.5, 94, 92]
    acc_thresholds = [99, 96, 94, 92]
    batch_sizes = [128, 128, 128, 128]
    
    for level in range(1,  ht.N_levels):
        
        valid_thresh = acc_thresholds[level]
        batch_sz = batch_sizes[level]
    
        valid_acc = 0    
        i = 0
        while valid_acc < valid_thresh:
            N_epoch = 400
            model_spec = {'k1': 5,  'k2': 5,  'd1': 32,  'd2':64 ,  'N_hidden':  128}   
            ht.train(level=level, acc_threshold=valid_thresh,
                         N_epoch=N_epoch, stop_threshold=False,
                         batch_sz=batch_sz,  model_spec=model_spec)
            if len(ht.current.records['validation_acc']) > 10:
                print("model_spec = ", ht.levels[level].model_spec)
                ht.plot('%s/training_record_level%d_Nepoch%d_d2%d_Nhidden%d_n%d.png' % (fig_dir, level,  N_epoch,  model_spec['d2'],  model_spec['N_hidden'],  i))
                valid_acc = ht.current.records['validation_acc'][-1]
                i += 1
            else:
                valid_acc = 0.0
    ht.close()
    
    
def find_good():
    ht = TrainHeadTrackerMRP()
    #acc = 0.0
    #while acc < 97:
#        acc = ht.train(level=0,  N_epoch=60)

#    acc = 0.0
  #  while acc < 99:
     #   acc = ht.train(level=1,  N_epoch=60)        
        
    acc = 0.0
    while acc < 99:
        acc = ht.train(level=2,  N_epoch=100)        
        
    acc = 0.0
    while acc < 99:
        acc = ht.train(level=3,  N_epoch=100) 


def search_models(level):
    """
    """   
    nrep=5
    data_fname = '/home/hjalmar/head_tracker/data/ht_data_overlap_mrp.hdf'
    fig_dir = '/home/hjalmar/head_tracker/figs/modelsearch'
    ht = TrainHeadTrackerMRP(data_fname=data_fname) 
    
    #all_params = {'k1': [7],  'k2': [7],  'd1': [ 16,  64],  'd2':[16,  64] ,  'Nhidden': [32, 64, 128]}
    all_params = {'d1': [ 16,  64],  'd2':[16,  64] ,  'Nhidden': [32, 64, 128,  256]}
      
    records = {}
    for param in all_params.keys():
        s = 'np.recarray(nrep,  dtype=['
        for value in all_params[param]:
            s += '( "' + str(value) + '", float), '
        s = s.rstrip(', ')
        s += '])'
        records[param] = eval(s)

    ht = TrainHeadTrackerMRP(data_fname=data_fname) 
    for param in all_params.keys():
        for value in all_params[param]:
                for i in range(nrep):
                    model_spec = {'k1': 5,  'k2': 5,  'd1': 32,  'd2':32 ,  'Nhidden':  64}
                    model_spec[param] = value
                    ht.train(level=level,  Nepoch=100, stop_threshold=False, model_spec=model_spec)
                    fname = ('%s/msearch_level%d_k1%d_k2%d_d1%d_d2%d_Nhidden%d_rep%d' % (fig_dir, level, model_spec['k1'], model_spec['k2'],  model_spec['d1'], model_spec['d2'], model_spec['Nhidden'],  i))
                    ht.plot(fname + '.png')
                    f_conf = open('%s.conf' % fname, 'w')
                    f_conf.write(ht.current.tostring())
                    f_conf.close()
                    records[param][str(value)][i] = ht.current.records['validation_acc'][-1]

    ht.close()
    return records


def some_extra(level):
    
    nrep = 5
    data_fname = '/home/hjalmar/head_tracker/data/ht_data_overlap_mrp.hdf'
    fig_dir = '/home/hjalmar/head_tracker/figs/modelsearch'
    ht = TrainHeadTrackerMRP(data_fname=data_fname) 
    for i in range(nrep):
        model_spec = {'k1': 5,  'k2': 5,  'd1': 32,  'd2':32 ,  'Nhidden':  256}
        ht.train(level=level,  Nepoch=100, stop_threshold=False, model_spec=model_spec)
        fname = ('%s/msearch_level%d_k1%d_k2%d_d1%d_d2%d_Nhidden%d_rep%d' % (fig_dir, level, model_spec['k1'], model_spec['k2'],  model_spec['d1'], model_spec['d2'], model_spec['Nhidden'],  i))
        ht.plot(fname + '.png')
        f_conf = open('%s.conf' % fname, 'w')
        f_conf.write(ht.current.tostring())
        f_conf.close()    


class levelClass:
    def __init__(self, level,  win_shape):
        self.level = level
        self.win_shape = win_shape
        self.win_nr = self.win_shape[0]
        self.win_nc = self.win_shape[1]
        self.win_sz = np.prod(self.win_shape)
        self.acc_threshold = None
        self.Nepoch = None
        self.Nclass = None
        self.Ntest = None
        self.frac_test = None
        self.Nvalid = None                
        self.frac_valid = None
        self.Ntrain = None                
        self.frac_train = None                
        self.batch_sz = None
        self.records = {'loss': [], 'epoch_acc': [],
                                 'validation_acc': [], 'test_acc': 0.0}


    def tostring(self):
        """
        """
        s = ('batch_sz: %d\tNdata: %d\tNepoch: %d\t'
        'Nclass: %d\tNtest: %d\t'
        'frac_test: %1.3f\tNvalid: %d\t'
        'frac_valid: %1.3f\tNtrain: %d\t'
        'frac_train: %1.3f' %  
        (self.batch_sz, self.Ndata, self.Nepoch,
         self.Nclass, self.Ntest, self.frac_test,
         self.Nvalid, self.frac_valid, self.Ntrain,
         self.frac_train))

        s = ('%s\nmodel_spec = {"k1": %d,  "k2": %d,  "d1": %d,  "d2": %d,  "Nhidden":  %d}' %
         (s,  self.model_spec['k1'],  self.model_spec['k2'],  self.model_spec['d1'],  self.model_spec['d2'],  self.model_spec['Nhidden']))
         
        s = '%s\nloss: %s' % (s, ' %1.3f,' * len(self.records['loss'])) % tuple(self.records['loss'])
        s = '%s\nepoch_acc: %s' % (s.rstrip(','), ' %1.3f,' * len(self.records['epoch_acc'])) % tuple(self.records['epoch_acc'])
        s = '%s\nvalidation_acc: %s' % (s.rstrip(','), ' %1.3f,' * len(self.records['validation_acc'])) % tuple(self.records['validation_acc'])
        s = '%s\ntest_acc: %1.3f' % (s.rstrip(','), self.records['test_acc'])
        
        return s


def HeSD(layer_shape):
    """
    Initialize weights according to He, Zang, Ren & Sun (2015).

    Parameters
    ----------------
    layer_shape  :  shape of the weight layer to initialize
    
    Returns
    -----------
    
    Usage
    ---------
    1st layer:
    shape = (patch_sz, patch_sz, 1, Nflt1)
    tf.Variable(tf.truncated_normal(shape, stddev=HeSD(shape)))

    Should be same as:
        tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FANIN', uniform=False)
    """
    nl = np.prod(layer_shape[:-1])
    return np.sqrt(2 / nl)


def weight_variable(shape):
    """
    """
    initial = tf.truncated_normal(shape, stddev=HeSD(shape))
    return tf.Variable(initial)


def bias_variable(shape):
    """
    """   
    return tf.Variable(tf.constant(0.1, shape=shape))    


def conv2d(x, w):
    """
    """ 
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    """     
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
                                         strides=[1, 2, 2, 1], padding='SAME')    


def create_graph_position(levelObj,  model_spec=None):
    """
    """
    
    if model_spec is None:
        if levelObj.level == 0:
            #k1,  k2 = 7,  5
            #d1,  d2= 16, 32
            #Nhidden=64           
            k1,  k2 = 5, 5
            d1,  d2= 64, 64
            Nhidden=128                       
        elif levelObj.level == 1:
            k1,  k2 = 5, 5
            d1,  d2= 64, 64
            Nhidden=128           
        else:
            k1,  k2 = 7,  5
            d1,  d2= 32, 32
            Nhidden=128
    else:
        k1,  k2 = model_spec['k1'],  model_spec['k2']
        d1,  d2= model_spec['d1'],  model_spec['d2']
        Nhidden = model_spec['Nhidden']
        
    levelObj.model_spec = {'k1': k1,  'k2': k2,  'd1': d1,  'd2':d2 ,  'Nhidden':  Nhidden}
    
    win_nr,  win_nc = levelObj.win_nr,  levelObj.win_nc
    Nclass = 2
    #Nbatches = levelObj.Nbatches
       
    class Model:
        
        def __init__(self):
            self.graph = tf.Graph()
            # Define ops and tensors in `g`.
            with self.graph.as_default():
                # Input data.
                self.X = tf.placeholder(tf.float32, shape=(None, win_nr, win_nc, 1))
                self.y_ = tf.placeholder(tf.float32, shape=(None))
                
                # Follows https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html
                # 2 convolutional layers
                # 1 fully connected
                # read out layer
                
                # TODO: best num filters, patch_sz (k), dropout on all layers or just last?
                
                # layer 1 -- convolutional
                w1 = weight_variable((k1, k1, 1, d1))  # k1 -- spatial filter size of layer 1;  d1 -- num filters of layer 1.
                b1 = bias_variable([d1])
                # Connect
                h_conv1 = tf.nn.relu(conv2d(self.X, w1) + b1)
                h_pool1 = max_pool_2x2(h_conv1)                

                # layer 2 -- convolutional
                w2 = weight_variable((k2, k2, d1, d2))
                b2 = bias_variable([d2])
                h_conv2 = tf.nn.relu(conv2d(h_pool1, w2) + b2)
                h_pool2 = max_pool_2x2(h_conv2)                

                # layer 3 -- fully connected with dropout
                self.keep_prob = tf.placeholder(tf.float32)
                h_pool2_shape = h_pool2.get_shape().as_list()
                w3 = weight_variable((np.prod(h_pool2_shape[1:]) , Nhidden))
                b3 = bias_variable([Nhidden]) 
                flat_shape = [-1, np.prod(h_pool2_shape[1:])]
                h_pool2_flat_drop = tf.nn.dropout(tf.reshape(h_pool2, flat_shape), self.keep_prob)
                h_fc1_drop = tf.nn.relu(tf.matmul(h_pool2_flat_drop, w3) + b3)
                # layer 4 -- readout layer with dropout
                h_fc2_drop = tf.nn.dropout(h_fc1_drop, self.keep_prob)
                w4 = weight_variable((Nhidden,  Nclass))
                b4 = bias_variable([Nclass]) 
                # LOGITS -- OK
                self.logits = tf.matmul(h_fc2_drop, w4) + b4   # Do softmax op when computing loss, supposedly better num stability.
                
                # Loss
                cross_entropy =tf.nn.sparse_softmax_cross_entropy_with_logits(
                    self.logits, tf.to_int64(self.y_), name='xentropy')
                self.loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
                #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y_))
                #self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)        
                # SOFTMAX -- CRASH
                #self.y = tf.nn.softmax(self.logits)
                correct = tf.equal(tf.argmax(self.logits, 1), tf.cast(self.y_,  tf.int64))
                self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                #correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y_, 1))
                #self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                #acc = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                #(acc * 100) / predictions.shape[0]
                self.saver = tf.train.Saver()
                
    return Model()


def inference(images, levelObj,  train=False,  model_spec=None):
    """
    """
    
    if model_spec is None:
        if levelObj.level == 0:
            k1,  k2 = 7,  5
            d1,  d2= 16, 32
            Nhidden=64            
        elif levelObj.level == 1:
            k1,  k2 = 5, 5
            d1,  d2= 64, 64
            Nhidden=128
        else:
            k1,  k2 = 7,  5
            d1,  d2= 32, 32
            Nhidden=128
    else:
        k1,  k2 = model_spec['k1'],  model_spec['k2']
        d1,  d2= model_spec['d1'],  model_spec['d2']
        Nhidden = model_spec['Nhidden']
        
    levelObj.model_spec = {'k1': k1,  'k2': k2,  'd1': d1,  'd2':d2 ,  'Nhidden':  Nhidden}
    
    Nclass = 2
        
    # layer 1 -- convolutional
    w1 = weight_variable((k1, k1, 1, d1))  # k1 -- spatial filter size of layer 1;  d1 -- num filters of layer 1.
    b1 = bias_variable([d1])
    # Connect
    h_conv1 = tf.nn.relu(conv2d(images, w1) + b1)
    h_pool1 = max_pool_2x2(h_conv1)                

    # layer 2 -- convolutional
    w2 = weight_variable((k2, k2, d1, d2))
    b2 = bias_variable([d2])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w2) + b2)
    h_pool2 = max_pool_2x2(h_conv2)                

    # layer 3 -- fully connected with dropout
    if train:
        keep_prob = 0.5
    else:
        keep_prob = 1.0
    
    h_pool2_shape = h_pool2.get_shape().as_list()
    w3 = weight_variable((np.prod(h_pool2_shape[1:]) , Nhidden))
    b3 = bias_variable([Nhidden]) 
    flat_shape = [-1, np.prod(h_pool2_shape[1:])]
    h_pool2_flat_drop = tf.nn.dropout(tf.reshape(h_pool2, flat_shape), keep_prob)
    h_fc1_drop = tf.nn.relu(tf.matmul(h_pool2_flat_drop, w3) + b3)
    # layer 4 -- readout layer with dropout
    h_fc2_drop = tf.nn.dropout(h_fc1_drop, keep_prob)
    w4 = weight_variable((Nhidden,  Nclass))
    b4 = bias_variable([Nclass]) 
    # logits
    logits = tf.matmul(h_fc2_drop, w4) + b4   # Do softmax op when computing loss, supposedly better num stability.
    return logits


def get_loss(logits,  labels):
    """
    """
    #labels = tf.to_float(labels)
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss
        
        
def training(loss, learning_rate):
    """
    """
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
       
    
def create_graph_rotation():
    """
    """
    
    
def softmax(logits):
    """
    Temporary replacement for tf.nn.softmax()
    See issue #2810 on github.com/tensorflow/tensorflow/issues  
    """
    e = np.exp(logits)
    return e / e.sum(axis=1,  keepdims=True)


def read_msearchfiles(level,  file_dir=None):
    
    if file_dir is None:
        file_dir = '/home/hjalmar/head_tracker/figs/modelsearch'
    file_dir = file_dir.rstrip('/')

    nrep = 5
    
    #all_params = {'k1': [3, 5, 7],  'k2': [3, 5, 7],  'd1': [4, 8, 16, 32],  'd2':[4, 8, 16, 32] ,  'Nhidden': [32, 64, 128]}
    all_params = {'k1': [5,  7],  'k2': [5,  7],  'd1': [ 16, 32,   64],  'd2':[16,  32,  64] ,  'Nhidden': [32, 64, 128,  256]}

    records = {}
    for param in all_params.keys():
        s = 'np.recarray(nrep,  dtype=['
        for value in all_params[param]:
            s += '( "' + str(value) + '", float), '
        s = s.rstrip(', ')
        s += '])'
        records[param] = eval(s)
    for key in records: 
        records[key][:] = np.nan
    
    n = 0
    for param in all_params.keys():
        for value in all_params[param]:
                for i in range(nrep):
                    #model_spec = {'k1': 5,  'k2': 5,  'd1': 16,  'd2':16 ,  'Nhidden':  64}
                    model_spec = {'k1': 5,  'k2': 5,  'd1': 32,  'd2':32 ,  'Nhidden':  64}
                    model_spec[param] = value
                    fname = ('%s/msearch_level%d_k1%d_k2%d_d1%d_d2%d_Nhidden%d_rep%d.conf' % (file_dir, level, model_spec['k1'], model_spec['k2'],  model_spec['d1'], model_spec['d2'], model_spec['Nhidden'],  i))
                    fn = glob(fname)
                    if len(fn) == 1:
                        n += 1
                        f = open(fn[0], 'r')
                        records[param][str(value)][i] = float(f.read()[-23:-17])
                        f.close()
    
    fnames = glob(file_dir + '/msearch_level%d_k*.conf' % level)
    n = len(fnames)

    return records,  n
    
def plot_msearch(records,  fig_fname=None):
    """
    """
    
    fig = plt.figure(figsize=[8,  9.5])
    m,  se = [],  []
    ax = fig.add_subplot(321)
    for dt in records['k1'].dtype.names:
        r = records['k1'][dt]
        m.append(np.nanmean(r))
        se.append(np.nanstd(r)/np.sqrt((~ np.isnan(r)).sum()))
        if ~ np.isnan(r).all():
            ax.plot([int(dt)]*(~ np.isnan(r)).sum(),  r[~ np.isnan(r)],  '.k',  color=[0.6]*3)        
    m,  se = np.array(m),  np.array(se)
    xstr = records['k1'].dtype.names
    x = np.zeros(len(xstr),  dtype=int)
    for i,  s in enumerate(xstr): x[i] = int(s)
    ok = ~np.isnan(m)
    ax.errorbar(x[ok],  m[ok],  yerr=se[ok],  marker='o',  mfc='r',  ms=10,  c='k')
    ax.set_ylabel('acc')
    ax.set_xlabel('k1')
    ax.set_xlim([x[ok].min()-0.5,  x[ok].max()+0.5])
    ax.set_xticks(x[ok])
    
    m,  se = [],  []
    ax = fig.add_subplot(322)
    for dt in records['k2'].dtype.names:
        r = records['k2'][dt]
        m.append(np.nanmean(r))
        se.append(np.nanstd(r)/np.sqrt((~ np.isnan(r)).sum()))
        if ~ np.isnan(r).all():
            ax.plot([int(dt)]*(~ np.isnan(r)).sum(),  r[~ np.isnan(r)],  '.k',  color=[0.6]*3)        
    m,  se = np.array(m),  np.array(se)
    xstr = records['k2'].dtype.names
    x = np.zeros(len(xstr),  dtype=int)
    for i,  s in enumerate(xstr): x[i] = int(s)
    ok = ~np.isnan(m)
    ax.errorbar(x[ok],  m[ok],  yerr=se[ok],  marker='o',  mfc='r',  ms=10,  c='k')
    ax.set_xlabel('k2')
    ax.set_xlim([x[ok].min()-0.5,  x[ok].max()+0.5])    
    ax.set_xticks(x[ok])
    
    m,  se = [],  []
    ax = fig.add_subplot(323)
    for dt in records['d1'].dtype.names:
        r = records['d1'][dt]
        m.append(np.nanmean(r))
        se.append(np.nanstd(r)/np.sqrt((~ np.isnan(r)).sum()))
        if ~ np.isnan(r).all():
            ax.plot([int(dt)]*(~ np.isnan(r)).sum(),  r[~ np.isnan(r)],  '.k',  color=[0.6]*3)        
    m,  se = np.array(m),  np.array(se)
    xstr = records['d1'].dtype.names
    x = np.zeros(len(xstr),  dtype=int)
    for i,  s in enumerate(xstr): x[i] = int(s)
    ok = ~np.isnan(m)
    ax.errorbar(x[ok],  m[ok],  yerr=se[ok],  marker='o',  mfc='r',  ms=10,  c='k')
    ax.set_ylabel('acc')
    ax.set_xlabel('d1')
    ax.set_xticks(x[ok]) 

    m,  se = [],  []
    ax = fig.add_subplot(324)
    for dt in records['d2'].dtype.names:
        r = records['d2'][dt]
        m.append(np.nanmean(r))
        se.append(np.nanstd(r)/np.sqrt((~ np.isnan(r)).sum()))
        if ~ np.isnan(r).all():
            ax.plot([int(dt)]*(~ np.isnan(r)).sum(),  r[~ np.isnan(r)],  '.k',  color=[0.6]*3)        
    m,  se = np.array(m),  np.array(se)
    xstr = records['d2'].dtype.names
    x = np.zeros(len(xstr),  dtype=int)
    for i,  s in enumerate(xstr): x[i] = int(s)
    ok = ~np.isnan(m)
    ax.errorbar(x[ok],  m[ok],  yerr=se[ok],  marker='o',  mfc='r',  ms=10,  c='k')
    ax.set_xlabel('d2')
    ax.set_xticks(x[ok]) 
    
    m,  se = [],  []
    ax = fig.add_subplot(325)
    for dt in records['Nhidden'].dtype.names:
        r = records['Nhidden'][dt]
        m.append(np.nanmean(r))
        se.append(np.nanstd(r)/np.sqrt((~ np.isnan(r)).sum()))
        if ~ np.isnan(r).all():
            ax.plot([int(dt)]*(~ np.isnan(r)).sum(),  r[~ np.isnan(r)],  '.k',  color=[0.6]*3)        
    m,  se = np.array(m),  np.array(se)
    xstr = records['Nhidden'].dtype.names
    x = np.zeros(len(xstr),  dtype=int)
    for i,  s in enumerate(xstr): x[i] = int(s)
    ok = ~np.isnan(m)
    ax.errorbar(x[ok],  m[ok],  yerr=se[ok],  marker='o',  mfc='r',  ms=10,  c='k')
    ax.set_xlabel('Nhidden')
    ax.set_xticks(x[ok]) 
    
    fig.suptitle('Level %d' % 1)
    
    if not fig_fname is None:
        fig.savefig(fig_fname)
        plt.close(fig)
    
class HeadTrackerMRP:
    """
    """
    def __init__(self, model_dir=None):
    
        if model_dir is None:
                model_dir = '/home/hjalmar/head_tracker/model/good'
        self.model_dir = model_dir.rstrip('/')
        
        self.frame = None

        self.mrp = MultiResolutionPyramid()
        #self.positon = np.recarray(self.mrp.Nlevels,  [('x',  float),  ('y',  float)])
        
    def track2video(self,  in_fname,  out_fname,  positions):
        """
        """
        if not tf.gfile.Exists(in_fname):
            raise ValueError('Failed to find file: %s' % in_fname)
            
        fst = FrameStepper(in_fname)        
        fps = int(round(1/fst.dt))
        FFMpegWriter = manimation.writers['ffmpeg']
        ttl = 'Head position tracking from video %s.'  % in_fname.split('/')[-1]
        metadata = dict(title=ttl, artist='Matplotlib',
                comment='more info...')  # TODO!
        writer = FFMpegWriter(fps=fps, metadata=metadata,  codec=None)  # TODO: set a good codec
        dpi = 96
        figsize = (fst.frame.shape[1]/dpi, fst.frame.shape[0]/dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)  # TODO dpi depends on the monitor used, remove this dependence
                                                                         # see: http://stackoverflow.com/questions/13714454/specifying-and-saving-a-figure-with-exact-size-in-pixels 
        with writer.saving(fig, out_fname, dpi):
            for pos in positions:
                fst.read_t(pos.t)
                ax = fig.add_subplot(111)
                ax.imshow(fst.frame,  origin='lower')
                ax.plot( pos.x,  pos.y,  'xr',  ms=20,  mew=2)
                ax.set_xlim([0,  fst.frame.shape[1]])
                ax.set_ylim([0,  fst.frame.shape[0]]) 
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_position([0,  0,  1,  1])
                writer.grab_frame()
                fig.clf()

        fst.close()
                
    def track_head(self,  video_fname,  t_start=0.0,  t_end=-1,  dur=None,  Nlevel=3):
        """
        """
        if not tf.gfile.Exists(video_fname):
            raise ValueError('Failed to find file: %s' % video_fname)
            
        scale_factor = 32/200   # TODO
        fst = FrameStepper(video_fname)
        
        if t_start < 0:
            raise ValueError('t_start cannot be less than 0.0 (beginning of the video).')
        if t_end < 0:
            t_end = fst.duration
        if not dur is None:
            t_end = min(t_end,  t_start + dur)
        if t_end > fst.duration:
            raise ValueError('t_end cannot be later %1.3f (time of the last frame)' %
                                        fst.duration)
        
        Nframe = int(((t_end - t_start) / fst.dt))
        est_pos = np.recarray(shape=Nframe,  dtype=[('t',  float),  ('x',  float), ('y',  float)]) 

        if Nframe > fst.tot_n:
            raise ValueError('Nframes cannot be greater than the number of frames video.')
        
        i = 0
        fst.read_t(t_start)
        while fst.t < t_end:
            frame = imresize(fst.frame.mean(axis=2), scale_factor)
            x,  y = self.predict_position(frame,  Nlevel,  verbose=False)
            est_pos[i].x = x / scale_factor
            est_pos[i].y = y / scale_factor
            est_pos[i].t = fst.t
            i += 1
            try:
                fst.next()
            except:
                import pdb
                pdb.set_trace()            
            
        fst.close()
            
        return est_pos
        
        
    def test_track_head(self,  log_fname,  video_dir,  Nlevel=2,  Nframe=300):
        """
        """
        scale_factor = 32/200   # TODO
        log_data, log_header = read_log_data(log_fname)
        video_fname = '%s/%s' % (video_dir.rstrip('/'), log_header['video_fname'])
        video_fname = glob(video_fname)[0]
        fst = FrameStepper(video_fname)
        
        est_pos = np.recarray(shape=Nframe,  dtype=[('x',  float), ('y',  float)])
        true_pos = np.recarray(shape=Nframe,  dtype=[('x',  float), ('y',  float)])
        
        if Nframe >= len(log_data):
            raise ValueError('Nframes cannot be greater than the number of frames in the log file.')
        
        for i, dat in enumerate(log_data[:Nframe]):

            print(Nframe-i)
            # Read the frame
            fst.read_t(dat['frame_time'])
            frame = imresize(fst.frame.mean(axis=2), scale_factor)
            # True head position
            true_pos.x[i] = dat['center_x']
            true_pos.y[i] = dat['center_y']
            
            x,  y = self.predict_position(frame,  Nlevel,  verbose=True)
            est_pos.x[i] = x / scale_factor
            est_pos.y[i] = y / scale_factor
            
        error = np.sqrt((true_pos.x - est_pos.x)**2 + (true_pos.x - est_pos.y))
        
        fst.close()

        return est_pos,  true_pos,  error
            
            
    def predict_position(self, frame,  Nlevel,  verbose=True):
        """
        """
        # TODO check frame shape and possibly reshape
        self.frame = frame
        
        self._position = np.recarray(Nlevel,  dtype=[('level',  int),  ('x',  float), ('y',  float)])
        self._w_position = np.recarray(Nlevel,  dtype=[('level',  int),  ('x',  float), ('y',  float)])
        prev_x,  prev_y = None,  None
        
        for level in self.mrp.levels[:Nlevel]:
            
            w_pos = self._w_position[level.level]
            w_pos.level = level.level
            pos = self._position[level.level]
            pos.level = level.level
            self.restore_position_model(level,  verbose=verbose)
            
            model = self.models_position[level.level]
            
            if level.level == 0:
                self.mrp.start(self.frame)
            else:
                #self.mrp.next((pos.y, pos.x))
                self.mrp.next((prev_y,  prev_x))
            
            win_sz = level.win_nc
            wins = level.wins[level.valid].reshape((-1, win_sz, win_sz, 1))
            feed_dict={model.X: wins, model.keep_prob: 1.0}
            prediction = softmax(model.logits.eval(session=model.session,  feed_dict=feed_dict))
            #self._predictions = prediction[:,  1] / prediction[:, 0]
            level._predictions = prediction[:,  1]
            #pos.y,  pos.x = level.centers[np.argmax(prediction[:,  1] / prediction[:, 0],  axis = 0),  :] 
            centers = level.centers[level.valid]
            pos.y,  pos.x = centers[np.argmax(level._predictions),  :]
            level.head_position['y'],  level.head_position['x'] = pos.y,  pos.x

            # weighted average position
            w = level._predictions/level._predictions.sum()
            w_pos.y = (centers[:,  0] * w).sum()
            w_pos.x = (centers[:,  1] * w).sum()
            prev_x,  prev_y = w_pos.x,  w_pos.y

        return w_pos.x,  w_pos.y
            
        
    def predict_rotation(self, X, model_fname):
        """
        X     - shape = (n_samples, 32, 32)
        """
        
        
    def restore_position_model(self,  level,  verbose=True):
        """
        """
                
        if not hasattr(self,  'models_position'):
            self.models_position = []
        if len(self.models_position) > level.level:
            msg = ('Model %s already restored.' % 
                         self.models_position[level.level].fname.split('/')[-1])
        else:
            
            if level.level == 0:
                model_spec = {'k1': 5,  'k2': 5,  'd1': 64,  'd2':64 ,  'Nhidden':  128}
            elif level.level == 1:
                model_spec = {'k1': 5,  'k2': 5,  'd1': 64,  'd2':64 ,  'Nhidden':  128}
            else:
                model_spec = {'k1': 7,  'k2': 5,  'd1': 32,  'd2':32 ,  'Nhidden':  128}            
            
            model = create_graph_position(level,  model_spec=model_spec)
            model_fn = '%s/position_level%d_acc*.ckpt*' % (self.model_dir,  level.level)
            model_fn = glob(model_fn)
            model_fn.sort()
            model.fname = model_fn[-1].rstrip('.meta')

            model.session = tf.Session(graph=model.graph)
            # Restore variables from disk.
            model.saver.restore(model.session, model.fname)
            self.models_position.append(model)
            msg = ('Model %s restored.' % 
                       self.models_position[-1].fname.split('/')[-1])
                       
        if verbose:
            print(msg)
    
    
    def plot(self,  true_pos = None,  fname=None):
    
        Nlevel = self.mrp.level_done + 1
        
        fig = plt.figure(figsize=[23, 6*Nlevel])
        for i,  level in enumerate(self.mrp.levels[:Nlevel]):
            win_sz = level.win_nr
            ax0 = fig.add_subplot(Nlevel, 3,  i*3+1) # (131)
            ax1 = fig.add_subplot(Nlevel, 3, i*3+2)            
            ax_x0,  ax_y0 = 0.65,  0.82-i*0.28
                
            ax0.imshow(self.frame,  origin='lower', cmap=plt.cm.gray)
            centers = level.centers
            predictions = level.predictions.copy()
            predictions[predictions<1E-18] = 1E-18
            scale = np.sqrt(predictions/predictions.max())
            if not true_pos is None:
                ax0.plot(true_pos['x'],  true_pos['y'],  'x',  mew=2,  ms=16,  mec=[1,  0.8431,  0])
                
            for j,  c in enumerate(centers):
                ax0.plot(c[ 1],  c[ 0],  'or',  mew=1,  ms=20*scale[j],  mfc='none',  mec='r')        
                txt_x,  txt_y = c[1]+1+scale[j],  c[0]+1+scale[j]
                ax0.text(txt_x,  txt_y,  str(j),  color=[0.3, 0.3, 0.9])
            best_x = level.centers[level.predictions.argmax(),  1]
            best_y = level.centers[level.predictions.argmax(),  0]            
            ax0.plot(best_x,  best_y,  'sg',  ms=20,  mew=2,  mfc='none',  mec='g')
            ax0.add_patch(patches.Rectangle((best_x-win_sz/2, best_y-win_sz/2), win_sz, win_sz, fill=False, ls='dotted', ec='g'))
            ax0.set_xlim([0,  self.frame.shape[1]])
            ax0.set_ylim([0,  self.frame.shape[0]])        
            # weighted average position
            w_x,  w_y = self.position[i].x,  self.position[i].y
            ax0.plot(w_x,  w_y,  'xg',  mew=2,  ms=20)
            
            ax1.plot(level.predictions,  '--k')
            ax1.plot(level.predictions,  'or')
            ax1.set_ylim([0,  1])
                    
            nwin = level.wins.shape[0]
            wdt = 0.09
            dx = 0.071
            if i == 0:
                axpos = [(ax_x0,  ax_y0,  wdt, wdt),  (ax_x0+dx,  ax_y0,  wdt, wdt),  (ax_x0+dx*2,   ax_y0,  wdt, wdt),  (ax_x0+dx*3,   ax_y0,  wdt, wdt), 
                                (ax_x0,  ax_y0-wdt,  wdt, wdt),  (ax_x0+dx,  ax_y0-wdt,  wdt, wdt),  (ax_x0+dx*2,  ax_y0-wdt,  wdt, wdt),  (ax_x0+dx*3,  ax_y0-wdt,  wdt, wdt), 
                                (ax_x0,  ax_y0-wdt*2,  wdt, wdt),  (ax_x0+dx,  ax_y0-wdt*2,  wdt, wdt),  (ax_x0+dx*2,  ax_y0-wdt*2,  wdt, wdt),  (ax_x0+dx*3,  ax_y0-wdt*2,  wdt, wdt)]
            else:
                axpos = [(ax_x0,  ax_y0,  wdt, wdt),  (ax_x0+dx,  ax_y0,  wdt, wdt),  (ax_x0+dx*2,   ax_y0,  wdt, wdt), 
                                (ax_x0,  ax_y0-wdt,  wdt, wdt),  (ax_x0+dx,  ax_y0-wdt,  wdt, wdt),  (ax_x0+dx*2,  ax_y0-wdt,  wdt, wdt), 
                                (ax_x0,  ax_y0-wdt*2,  wdt, wdt),  (ax_x0+dx,  ax_y0-wdt*2,  wdt, wdt),  (ax_x0+dx*2,  ax_y0-wdt*2,  wdt, wdt)]
                
            for k in range(nwin):
                ax = fig.add_axes(axpos[k])
                ax.imshow(level.wins[k, :, :],  origin='lower', cmap=plt.cm.gray,  clim=(-2,  2))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(2, level.wins.shape[1]-2, '%1.3f' % level.predictions[k],  va='top',  ha='left',  color='r')        
                ax.plot([0,  level.predictions[k]* level.wins.shape[2]], [0,  0],  '-g',  lw=4)
                
        if not fname is None:
            try:
                fig.savefig(fname)
            except:
                import pdb
                pdb.set_trace()
            plt.close(fig)  
  
        fig = plt.figure()  
        ax = fig.add_subplot(111)
        frame = self.frame.copy()
        frame = frame-frame.mean()
        frame = frame/frame.std()
        ax.imshow(frame,  origin='lower', cmap=plt.cm.gray,  clim=(-2,  2))
        ax.plot(self.position['x'][Nlevel-1],  self.position['y'][Nlevel-1],  'xr',  ms=20,  mew=2)
        ax.set_xlim([0,  self.frame.shape[1]])
        ax.set_ylim([0,  self.frame.shape[0]]) 
        ax.set_xticks([])
        ax.set_yticks([])
        if not fname is None:
            fname = fname.split('.png')[0] + 'simple.png'
            fig.savefig(fname)
            plt.close(fig)  
    


    def close(self):
        """
        """
        for model in self.models_position:
            model.session.close()

        
            
#def test_head_detect():
#    
    #video_fname = '/media/hjalmar/VitaminD/Sound/ICe/ATrain/continous/video/AT_20150206_1272_01_09-15-58_cont.mp4'

#    model_fname = '/home/hjalmar/head-tracker_data/models/head_detect/head_detect_model_neg11_accd99.3.ckpt'
#    
#    nframes = 100
#    scale_factor = 32 / 200
#    ht = HeadTracker(train_data_fname=dat_fname)
#    
#    fstp = FrameStepper(video_fname)
#    fstp.read_t(1.5)
#    frame = imresize(fstp.frame.mean(axis=2), scale_factor)
#    
#    fig = plt.figure()
#    r0, c0 = 0.0, 0.0
#    r1, c1 = frame.shape
#    for i in range(nframes):
#        
#        preds = []
#        XY = []
#        thresh = 1.0
#        
#
#        for x in range(-32//2, 32//2, 2):
#            for y in range(-32//2, 32//2, 2):
#                X, RC = window_image(frame, (32,32), (y, x), 'pad', 'white')
#                X = whiten(X)
#
#                P = ht.predict(X.reshape((-1,32,32,1)), model_fname)
#                if (P[:,1] >= thresh).any():
#                    ix = P[:,1] >= thresh
#                    preds.extend(P[ix,1])
#                    for rc in RC[ix,:]:
#                        XY.append([rc[1], rc[0]])
#                        
#        XY = np.array(XY)
#        idx = XY[:, 0] > 0
#        idx = np.logical_and(idx, XY[:, 1] > 0)
#        idx = np.logical_and(idx, XY[:, 0] < frame.shape[1])
#        idx = np.logical_and(idx, XY[:, 1] < frame.shape[0])
#        XY = XY[idx, :]
#        XY[:,0] += int(c0)
#        XY[:,1] += int(r0)
#        clust = cluster.hierarchy.fclusterdata(XY,0.5)
#
#        max_clust = XY[np.bincount(clust).argmax()==clust, :]
#        clust_median = np.median(max_clust, axis=0)
#        
#        ax = fig.add_subplot(111)
#        ax.imshow(fstp.frame)
#        #ax.plot([c0/scale_factor, c1/scale_factor], [r0/scale_factor, r0/scale_factor], '-g', lw=2)
#        #ax.plot([c0/scale_factor, c1/scale_factor], [r1/scale_factor, r1/scale_factor], '-g', lw=2)
#        #ax.plot([c0/scale_factor, c0/scale_factor], [r0/scale_factor, r1/scale_factor], '-g', lw=2)
#        #ax.plot([c1/scale_factor, c1/scale_factor], [r0/scale_factor, r1/scale_factor], '-g', lw=2)
#        ax.set_xticks([])
#        ax.set_yticks([])
#        for xy in XY:
#            ax.plot(xy[0]/scale_factor, xy[1]/scale_factor, 'ok')
#        ax.plot(clust_median[0]/scale_factor, clust_median[1]/scale_factor, '+r', ms=20)
#        ax.set_xlim([0, fstp.frame.shape[1]])
#        ax.set_ylim([0, fstp.frame.shape[0]])
#        fig.savefig('/home/hjalmar/head-tracker_data/fig/head_tracking_%03d.png' % i)
#        fig.clf()
#        
#        r_off, c_off = clust_median[1], clust_median[0]
#        fstp.next()
#        frame = imresize(fstp.frame.mean(axis=2), scale_factor)
#        r0, r1 = max(r_off-36,0), min(r_off+36,frame.shape[0])
#        c0, c1 = max(c_off-36,0), min(c_off+36,frame.shape[1])
#        #frame = frame[r0:r1, c0:c1]
#
#    
#    ht.close()

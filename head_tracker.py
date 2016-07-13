# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:17:35 2016

@author: hjalmar
"""

import tensorflow as tf
#from ht_helper import FrameStepper  # Import has to come after tensorflow bc bug/crash
from ht_helper import MultiResolutionPyramid, angle2class,  FrameStepper,  class2angle
from ht_helper import get_window,  angles2complex,  complex2angles,  get_gaze_line
from data_preparation import read_log_data
import numpy as np
import re
from scipy.misc import imresize
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as manimation
#import time

class TrainPositionModel:
    """
    """
    def __init__(self, data_dir=None,  model_dir=None):
    
        if data_dir is None:
            data_dir = '/home/hjalmar/head_tracker/data'
        self.data_dir = data_dir.rstrip('/')            
        if model_dir is None:
            model_dir = '/home/hjalmar/head_tracker/model'
        self.model_dir = model_dir.rstrip('/')
        
        self.batch_sz = 128
        self.Nclass = 2
        self.Nlevels = 4
        self.level = -1
        self.levels = []
        self.current = None
        self.data = None
        self.labels = None
        win_shapes = [(48, 48),  (32, 32),  (32, 32),  (32, 32)]
                
        for k in range(self.Nlevels):
            self.levels.append(levelClass(k,  win_shapes[k]))
            
    
#    def get_valid_data(self,  valid_fname):
#        
#        images,  labels = self.get_inputs(valid_fname,  train=False)
#        
#        Nvalid = 30000
#        batch_sz = 128
#        valid_images,  valid_labels = [],  []
#        Niter = int(np.ceil(Nvalid / batch_sz))
#        
#        with tf.Session() as sess:
#            
#            sess.run(tf.initialize_all_variables())
#            coord = tf.train.Coordinator()
#            threads = tf.train.start_queue_runners(sess=sess,  coord=coord)   
#            
#            try:               
#                i = 0
#                while i < Niter and not coord.should_stop():
#                    imgs,  lbls = sess.run([images,  labels])            
#                    valid_images.extend(imgs)
#                    valid_labels.extend(lbls)
#                    i += 1   
#            except Exception as e: 
#                coord.request_stop(e)
#                    
#            coord.request_stop()
#            coord.join(threads, stop_grace_period_secs=10)
#            
#        return np.array(valid_images),  np.array(valid_labels)
        
            
    def get_inputs(self, fname, N, train=False, flip=True):
        """
        N  - Ntrain or Nvalid: number_of_examples_per_epoch
        """
        if not tf.gfile.Exists(fname):
            raise ValueError('Failed to find file: %s' % fname)
        
        if train:
            Nepoch = self.current.Nepoch
        else:
            Nepoch = 1
            
        with tf.name_scope('input'):
            fname_queue = tf.train.string_input_producer(
                [fname], num_epochs=Nepoch)

            # Even when reading in multiple threads, share the filename
            # queue.
            image, label = self._read_and_decode(fname_queue)
            #print(image)
            if train:
                # Distort image
                image = self._distort_inputs(image,  flip=flip)
                n_threads = 8
            else:
                n_threads = 4
                
            # Shuffle the examples and collect them into batch_sz batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            # Ensures a minimum amount of shuffling of examples.
                
            min_queue_examples = int(N * 0.4)
            
            images, labels = tf.train.shuffle_batch([image, label],
                                                    batch_size=self.batch_sz,
                                                    num_threads=n_threads,
                                                    capacity=min_queue_examples + 3 * self.batch_sz,
                                                    min_after_dequeue=min_queue_examples)
                                                                      
            return images, labels            


#==============================================================================
#     def evaluate(self,  valid_fname,  fname_ckpt,  model_spec=None,  batch_sz=128):
#         
#         self.current.Nclass = 2
#         
#           #self.session = tf.Session(graph=self.model.graph)
#         # Restore variables from disk.
#         # self.model.saver.restore(self.session, self.model.fname)
#         
#         with tf.Graph().as_default():
# 
#             images,  labels = self.get_inputs(valid_fname,  train=False)
#             logits = inference(images,  self.current,  train=False, model_spec=model_spec) 
#             correct = tf.equal(tf.argmax(logits, 1), tf.cast(labels,  tf.int64))
#             accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#             saver = tf.train.Saver()
#             #saver = tf.train.Saver(tf.all_variables())
#             with tf.Session() as sess:
#                 sess.run(tf.initialize_all_variables())
#                 saver.restore(sess, fname_ckpt)
#                 
#                 #import pdb
#                 #pdb.set_trace()
#                 
#                 # Start the queue runners.
#                 acc = 0.0
#                 coord = tf.train.Coordinator()
#                 threads = tf.train.start_queue_runners(sess=sess,  coord=coord)               
#                 #print(sess.run(labels))
#                 try:
#                     num_iter = int(np.ceil(30000 / 128))
#                     step = 0
#                     #print(sess.run([correct]))
#                     while step < num_iter and not coord.should_stop():
#                         
#                         a = sess.run(accuracy)
#                         print(a)
#                         acc += a
#                         step += 1
#                         
#                 except Exception as e:  # pylint: disable=broad-except
#                     coord.request_stop(e)
# 
#                 coord.request_stop()
#                 coord.join(threads, stop_grace_period_secs=10)
#==============================================================================

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
        #y = (np.arange(self.Nclass) == y[:,None]).astype(np.float32)

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
           
           
    def train(self, level,  Nepoch,  model_spec=None):
        """
        """
        model_fname = '%s/position_level%d' % (self.model_dir,  level)
        train_fname = '%s/train_position_level%d_N*.tfrecords' % (self.data_dir,  level)
        valid_fname = '%s/dev_position_level%d_N*.tfrecords' % (self.data_dir,  level)
        train_fname = glob(train_fname)
        if not len(train_fname) == 1:
            raise ValueError('Something wrong with the file name of the training data.')
        else:
            train_fname = train_fname[0]
        valid_fname = glob(valid_fname)            
        if not len(valid_fname) == 1:
            raise ValueError('Something wrong with the file name of the validation data.')
        else:
            valid_fname = valid_fname[0]
            
        self.current = self.levels[level]
        if level == 0:
            flip = False
        else:
            flip = True
        #threshold = self.current.acc_threshold
               
        #records = self.current.records
        self.current.batch_sz = self.batch_sz
        self.current.Nepoch = Nepoch
        batch_sz = self.current.batch_sz
        Nvalid = int(re.search(r'[\d]{5,6}', valid_fname).group())
        self.current.Nvalid = Nvalid
        Ntrain = int(re.search(r'[\d]{5,6}', train_fname).group())
        self.current.Ntrain = Ntrain
        self.current.Nbatch_per_epoch = self.current.Ntrain // batch_sz
        #self.current.Nbatch = self.current.Nbatch_per_epoch * Nepoch
                
        # Get graph/model
        self.current.Nclass = self.Nclass
        #model = create_graph_position(self.current, model_spec=model_spec)
        learning_rate = 1e-4
                
        valid_X,  valid_y = [],  []
        
        model = create_graph_position(self.current, model_spec=model_spec)        
        
        with model.graph.as_default():
            # Input images and labels.
            images, labels = self.get_inputs(train_fname, Ntrain, train=True,  flip=flip)
            valid_images, valid_labels = self.get_inputs(valid_fname,  Nvalid,  train=False,  batch_sz=1000)        
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
                    while (epoch_n < Nepoch) and not coord.should_stop():
                        step += 1                        
                        # Train
                        X,  y = session.run([images,  labels])      
                        feed_dict = {model.X: X, model.y_: y, model.keep_prob: 0.5}
                        optimizer.run(feed_dict)
                        epoch_loss += model.loss.eval(feed_dict=feed_dict)
                        feed_dict[model.keep_prob] = 1.0
                        epoch_acc += model.accuracy.eval(feed_dict=feed_dict)
                        
                        if (step % self.current.Nbatch_per_epoch == 0):
                            epoch_n += 1
                            epoch_acc /= self.current.Nbatch_per_epoch
                            print('   %-5d|       %-12.3f|        %-14.2f'  %  (epoch_n, epoch_loss, epoch_acc*100))

                            if (epoch_n % 10 == 0):
                                v_acc,  i = 0.0,  0
                                if len(valid_y) < 1:
                                    load_valid = True
                                else:
                                    load_valid = False
                                while i < Nvalid // 1000:
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
                    while i < Nvalid // 1000:
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
        chance_acc = 100 * (self.current.Nnegative / self.current.Ndata)
        axl.plot([0,  len(self.current.records['epoch_acc'])],  [chance_acc,  chance_acc],  '--r')
        axl.set_ylim([5*np.floor(chance_acc/5), 100])
        axl.grid()
        axr.legend(loc=4, fontsize=10, frameon=False)
        axl.legend(loc=3, fontsize=10, frameon=False)

        if not fname is None:
            fig.savefig(fname)
            plt.close(fig)
            

class TrainOrientationModel:
    """
    """
    def __init__(self, Nclass=8,  data_dir=None,  model_dir=None):
    
        if data_dir is None:
            data_dir = '/home/hjalmar/head_tracker/data'
        self.data_dir = data_dir.rstrip('/')            
        if model_dir is None:
            model_dir = '/home/hjalmar/head_tracker/model'
        self.model_dir = model_dir.rstrip('/')
        
        self.Nclass = Nclass
        self.win_sz = 40
        self.batch_sz = 256
               
            
    def get_inputs(self, fname, Nepoch, Nex_per_epoch, batch_sz, train=False):
        """
        Nex_per_epoch  - Ntrain or Nvalid: number_of_examples_per_epoch
        """
        if not tf.gfile.Exists(fname):
            raise ValueError('Failed to find file: %s' % fname)
                    
        with tf.name_scope('input'):
            fname_queue = tf.train.string_input_producer(
                [fname], num_epochs=Nepoch)

            # Even when reading in multiple threads, share the filename
            # queue.
            image, label = self._read_and_decode(fname_queue)

            if train:
                # Distort image
                image = self._distort_inputs(image)
                n_threads = 8
            else:
                n_threads = 4
                
            # Shuffle the examples and collect them into batch_sz batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            # Ensures a minimum amount of shuffling of examples.             
            min_queue_examples = int(Nex_per_epoch * 0.4)
            
            images, labels = tf.train.shuffle_batch([image, label],
                                                    batch_size=self.batch_sz,
                                                    num_threads=n_threads,
                                                    capacity=min_queue_examples + 3 * self.batch_sz,
                                                    min_after_dequeue=min_queue_examples)
                                                                      
            return images, labels            

    
    def _read_and_decode(self,  fname_queue):
        reader = tf.TFRecordReader()
        _,  serialized_example = reader.read(fname_queue)
        features = tf.parse_single_example(
            serialized_example, 
            features={'image_raw': tf.FixedLenFeature([],  tf.string), 
                              'label': tf.FixedLenFeature([],  tf.int64)
                              })
  
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([self.win_sz *  self.win_sz])
        image = tf.reshape(image,  [self.win_sz,  self.win_sz,  1])
      
        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5        
        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['label'], tf.int32)

        return image, label
            
    
    def _distort_inputs(self,  image):
        """
        Don't flip orientation images
        """ 
        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        # Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_whitening(image)
        
        return image
           
           
    def train(self, Nepoch,  model_spec=None):
        """
        """
        model_fname = '%s/orientation' % (self.model_dir)
        train_fname = '%s/train_orientation_N*.tfrecords' % (self.data_dir)
        valid_fname = '%s/dev_orientation_N*.tfrecords' % (self.data_dir)
        train_fname = glob(train_fname)
        if not len(train_fname) == 1:
            raise ValueError('Something wrong with the file name of the training data.')
        else:
            train_fname = train_fname[0]
        valid_fname = glob(valid_fname)
        if not len(valid_fname) == 1:
            raise ValueError('Something wrong with the file name of the validation data.')
        else:
            valid_fname = valid_fname[0]
                           
        batch_sz = 256 #128
        Nvalid = int(re.search(r'[\d]{5,6}', valid_fname).group())
        Ntrain = int(re.search(r'[\d]{5,6}', train_fname).group())
        Nbatch_per_epoch = Ntrain // batch_sz
        #Nbatch = Nbatch_per_epoch * Nepoch
                
        Nclass = self.Nclass
        learning_rate = 1e-4
                
        valid_X,  valid_y = [],  []
        
        model = create_graph_orientation(self.win_sz,  Nclass,  model_spec=model_spec)        
        
        with model.graph.as_default():
            # Input images and labels.
            images, labels = self.get_inputs(train_fname,  Nepoch,  Ntrain,  batch_sz=batch_sz, train=True)
            valid_images, valid_labels = self.get_inputs(valid_fname, 1,  Nvalid,  batch_sz=1000,  train=False)
            saver = model.saver

            with tf.Session(graph=model.graph) as session:
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.loss) 
                session.run(tf.initialize_all_variables())
                # Start input enqueue threads
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=session,  coord=coord)  
      
                validation_accuracy = []
                train_accuracy = []                
                step, epoch_n  = 0,  0
                epoch_loss,  epoch_acc = 0.0,  0.0
                
                try:

                    while (epoch_n < Nepoch) and not coord.should_stop():
                        step += 1                        
                        # Train
                        X,  angles = session.run([images,  labels])  
                        y = angle2class(angles,  Nclass)   # angles are -180 to 180 needs to be range(Nclass)
                        feed_dict = {model.X: X, model.y_: y, model.keep_prob: 0.5}
                        optimizer.run(feed_dict)
                        epoch_loss += model.loss.eval(feed_dict=feed_dict)
                        feed_dict[model.keep_prob] = 1.0
                        epoch_acc += model.accuracy.eval(feed_dict=feed_dict)
                        
                        if (step % Nbatch_per_epoch == 0):
                            epoch_n += 1
                            epoch_acc /= Nbatch_per_epoch
                            print('   %-5d|       %-19.1f|        %-14.2f'  %  (epoch_n, epoch_loss, epoch_acc*100))

                            if (epoch_n % 5 == 0):
                                v_acc,  i = 0.0,  0
                                if len(valid_y) < 1:
                                    load_valid = True
                                else:
                                    load_valid = False
                                while i < Nvalid // 1000:
                                    if load_valid and not coord.should_stop():
                                        X,  angles = session.run([valid_images,  valid_labels]) 
                                        valid_X.append(X)
                                        valid_y.append(angle2class(angles,  Nclass))  # Bin 360 degrees into Nclass bins with values 0:Nclass-1.
                                    feed_dict = {model.X: valid_X[i], model.y_: valid_y[i], model.keep_prob: 1.0}
                                    v_acc += model.accuracy.eval(feed_dict=feed_dict)
                                    i += 1
                                validation_accuracy.append(100 * (v_acc/i))
                                train_accuracy.append(epoch_acc*100)                                
                                saver.save(session,
                                           '%s_Nclass%d_acc%1.1f.ckpt' % 
                                           (model_fname, Nclass, validation_accuracy[-1]),
                                           global_step=epoch_n)

                            epoch_loss,  epoch_acc = 0.0,  0.0                          
                        
                except tf.errors.OutOfRangeError:
                    print('Done training for %d epochs, %d steps.' % (epoch_n, step-1))       
                                            
                finally:
                    v_acc,  i = 0.0,  0
                    if len(valid_y) < 1:
                        load_valid = True
                    else:
                        load_valid = False
                    while i < Nvalid // 1000:
                        if load_valid and not coord.should_stop():
                            X,  y = session.run([valid_images,  valid_labels]) 
                            valid_X.append(X)
                            valid_y.append(np.int32(y / (360/Nclass)))  # Bin 360 degrees into Nclass bins with values 0:Nclass-1.
                        feed_dict = {model.X: valid_X[i], model.y_: valid_y[i], model.keep_prob: 1.0}
                        v_acc += model.accuracy.eval(feed_dict=feed_dict)
                        i += 1
                    validation_accuracy.append(100 * (v_acc/i))
                    saver.save(session, '%s_Nclass%d_acc%1.1f.ckpt' % 
                                        (model_fname, Nclass, validation_accuracy[-1]),
                               global_step=epoch_n)
                
                    # Ask threads to stop
                    coord.request_stop()

                # Wait for threads to finish.
                coord.join(threads)
                session.close() 
                
                print('Training accuracy:',  train_accuracy)
                print('Validation accuracy:',  validation_accuracy)
                
        return validation_accuracy[-1]
                                        
             
def train_all(valid_thresh=99):
    
    data_fname = '/home/hjalmar/head_tracker/data/ht_data_overlap_mrp.hdf'
    model_dir = '/home/hjalmar/head_tracker/model'
    fig_dir = '/home/hjalmar/head_tracker/figs'
    ht = TrainHeadTrackerMRP(model_dir=model_dir, data_fname=data_fname)
    
    #acc_thresholds = [98, 95.5, 94, 92]
    acc_thresholds = [99, 96, 94, 92]
    batch_sizes = [128, 128, 128, 128]
    
    for level in range(1,  ht.Nlevels):
        
        valid_thresh = acc_thresholds[level]
        batch_sz = batch_sizes[level]
    
        valid_acc = 0    
        i = 0
        while valid_acc < valid_thresh:
            Nepoch = 400
            model_spec = {'k1': 5,  'k2': 5,  'd1': 32,  'd2':64 ,  'Nhidden':  128}   
            ht.train(level=level, acc_threshold=valid_thresh,
                         Nepoch=Nepoch, stop_threshold=False,
                         batch_sz=batch_sz,  model_spec=model_spec)
            if len(ht.current.records['validation_acc']) > 10:
                print("model_spec = ", ht.levels[level].model_spec)
                ht.plot('%s/training_record_level%d_Nepoch%d_d2%d_Nhidden%d_n%d.png' % (fig_dir, level,  Nepoch,  model_spec['d2'],  model_spec['Nhidden'],  i))
                valid_acc = ht.current.records['validation_acc'][-1]
                i += 1
            else:
                valid_acc = 0.0
    ht.close()
    
    
def find_good():
    ht = TrainHeadTrackerMRP()
    #acc = 0.0
    #while acc < 97:
#        acc = ht.train(level=0,  Nepoch=60)

#    acc = 0.0
  #  while acc < 99:
     #   acc = ht.train(level=1,  Nepoch=60)        
        
#    acc = 0.0
 #   while acc < 99:
  #      acc = ht.train(level=2,  Nepoch=100)        
        
    acc = 0.0
    while acc < 99:
        acc = ht.train(level=3,  Nepoch=100) 

#==============================================================================
# 
# def search_models(level):
#     """
#     """   
#     nrep=5
#     data_fname = '/home/hjalmar/head_tracker/data/ht_data_overlap_mrp.hdf'
#     fig_dir = '/home/hjalmar/head_tracker/figs/modelsearch'
#     ht = TrainHeadTrackerMRP(data_fname=data_fname) 
#     
#     #all_params = {'k1': [7],  'k2': [7],  'd1': [ 16,  64],  'd2':[16,  64] ,  'Nhidden': [32, 64, 128]}
#     all_params = {'d1': [ 16,  64],  'd2':[16,  64] ,  'Nhidden': [32, 64, 128,  256]}
#       
#     records = {}
#     for param in all_params.keys():
#         s = 'np.recarray(nrep,  dtype=['
#         for value in all_params[param]:
#             s += '( "' + str(value) + '", float), '
#         s = s.rstrip(', ')
#         s += '])'
#         records[param] = eval(s)
# 
#     ht = TrainHeadTrackerMRP(data_fname=data_fname) 
#     for param in all_params.keys():
#         for value in all_params[param]:
#                 for i in range(nrep):
#                     model_spec = {'k1': 5,  'k2': 5,  'd1': 32,  'd2':32 ,  'Nhidden':  64}
#                     model_spec[param] = value
#                     ht.train(level=level,  Nepoch=100, stop_threshold=False, model_spec=model_spec)
#                     fname = ('%s/msearch_level%d_k1%d_k2%d_d1%d_d2%d_Nhidden%d_rep%d' % (fig_dir, level, model_spec['k1'], model_spec['k2'],  model_spec['d1'], model_spec['d2'], model_spec['Nhidden'],  i))
#                     ht.plot(fname + '.png')
#                     f_conf = open('%s.conf' % fname, 'w')
#                     f_conf.write(ht.current.tostring())
#                     f_conf.close()
#                     records[param][str(value)][i] = ht.current.records['validation_acc'][-1]
# 
#     ht.close()
#     return records
# 
# 
# def some_extra(level):
#     
#     nrep = 5
#     data_fname = '/home/hjalmar/head_tracker/data/ht_data_overlap_mrp.hdf'
#     fig_dir = '/home/hjalmar/head_tracker/figs/modelsearch'
#     ht = TrainHeadTrackerMRP(data_fname=data_fname) 
#     for i in range(nrep):
#         model_spec = {'k1': 5,  'k2': 5,  'd1': 32,  'd2':32 ,  'Nhidden':  256}
#         ht.train(level=level,  Nepoch=100, stop_threshold=False, model_spec=model_spec)
#         fname = ('%s/msearch_level%d_k1%d_k2%d_d1%d_d2%d_Nhidden%d_rep%d' % (fig_dir, level, model_spec['k1'], model_spec['k2'],  model_spec['d1'], model_spec['d2'], model_spec['Nhidden'],  i))
#         ht.plot(fname + '.png')
#         f_conf = open('%s.conf' % fname, 'w')
#        f_conf.write(ht.current.tostring())
#        f_conf.close()    
#==============================================================================



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

                                         
def max_pool_3x2(x):
    """
    """     
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], 
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
    #Nbatch = levelObj.Nbatch
       
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
       
    
def create_graph_orientation(win_sz,  Nclass,  model_spec=None):
    """
    """
    
    if model_spec is None:
            k1,  k2 = 5,  5
            d1,  d2= 64, 64
            Nhidden1=384
            Nhidden2=192
    else:
        k1,  k2 = model_spec['k1'],  model_spec['k2']
        d1,  d2= model_spec['d1'],  model_spec['d2']
        Nhidden = model_spec['Nhidden']
                   
    class Model:
        
        def __init__(self):
            self.graph = tf.Graph()
            # Define ops and tensors in `g`.
            with self.graph.as_default():
                # Input data.
                self.X = tf.placeholder(tf.float32, shape=(None, win_sz, win_sz, 1))
                self.y_ = tf.placeholder(tf.float32, shape=(None))
                                
                # layer 1 -- convolutional
                w1 = weight_variable((k1, k1, 1, d1))  # k1 -- spatial filter size of layer 1;  d1 -- num filters of layer 1.
                b1 = bias_variable([d1])
                # Connect
                h_conv1 = tf.nn.relu(conv2d(self.X, w1) + b1)
                h_pool1 = max_pool_3x2(h_conv1)    
                
                # norm1
                h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, 
                                                 beta=0.75, name='norm1')                

                # layer 2 -- convolutional
                w2 = weight_variable((k2, k2, d1, d2))
                b2 = bias_variable([d2])
                h_conv2 = tf.nn.relu(conv2d(h_norm1, w2) + b2)
                
                # norm2
                h_norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, 
                                                 beta=0.75, name='norm2')                
                h_pool2 = max_pool_3x2(h_norm2)                

                # layer 3 -- fully connected with dropout
                self.keep_prob = tf.placeholder(tf.float32)
                h_pool2_shape = h_pool2.get_shape().as_list()
                flat_shape = np.prod(h_pool2_shape[1:])
                w3 = weight_variable((flat_shape , Nhidden1))
                b3 = bias_variable([Nhidden1]) 
                flat_shape = [-1, flat_shape]
                h_pool2_flat_drop = tf.nn.dropout(tf.reshape(h_pool2, flat_shape), self.keep_prob)
                h_fc1_drop = tf.nn.relu(tf.matmul(h_pool2_flat_drop, w3) + b3)
                # layer 4 -- fully connected with dropout
                w4 = weight_variable((Nhidden1 , Nhidden2))
                b4 = bias_variable([Nhidden2]) 
                h_fc2_drop = tf.nn.relu(tf.matmul(tf.nn.dropout(h_fc1_drop, self.keep_prob),  w4) + b4)
                # layer 5 -- readout layer with dropout
                h_fc3_drop = tf.nn.dropout(h_fc2_drop, self.keep_prob)
                w5 = weight_variable((Nhidden2,  Nclass))
                b5 = bias_variable([Nclass]) 

                self.logits = tf.matmul(h_fc3_drop, w5) + b5   # Do softmax op when computing loss, supposedly better num stability.
                
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
    
class HeadTracker:
    """
    """
    def __init__(self, model_dir=None):
    
        if model_dir is None:
                model_dir = '/home/hjalmar/head_tracker/model/good'
        self.model_dir = model_dir.rstrip('/')
        
        self.frame = None

        self.mrp = MultiResolutionPyramid()
        #self.positon = np.recarray(self.mrp.Nlevels,  [('x',  float),  ('y',  float)])
        
        
    def track2video(self,  in_fname,  out_fname,  track):
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
            for trk in track:
                fst.read_t(trk.t)
                
                ax = fig.add_subplot(111)
                ax.imshow(fst.frame,  origin='lower')
                x,  y = trk.x,  trk.y
                ax.plot(x,  y,  'xr',  ms=20,  mew=2)
                gzl_x, gzl_y = get_gaze_line(trk.angle, x, y, 50, units='deg')
                ax.plot(gzl_x,  gzl_y,  '-g',  lw=2)
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
        verbose=False
        Nclass = 12
        scale_factor = 32/200   # TODO
        log_data, log_header = read_log_data(log_fname)
        video_fname = '%s/%s' % (video_dir.rstrip('/'), log_header['video_fname'])
        video_fname = glob(video_fname)[0]
        fst = FrameStepper(video_fname)
        
        est_track = np.recarray(shape=Nframe,  dtype=[('t',  float), 
                                                                                       ('x',  float),
                                                                                       ('y',  float), 
                                                                                       ('angle',  float)])
        true_track = np.recarray(shape=Nframe,  dtype=[('t',  float),
                                                                                         ('x',  float),
                                                                                         ('y',  float), 
                                                                                         ('angle',  float)])
                  
        if Nframe >= len(log_data):
            raise ValueError('Nframes cannot be greater than the number of frames in the log file.')
        
        for i, dat in enumerate(log_data[:Nframe]):

            print(Nframe-i)
            # Read the frame
            fst.read_t(dat['frame_time'])
            frame = imresize(fst.frame.mean(axis=2), scale_factor)
            
            # Time of frame
            true_track.t[i] = fst.t
            est_track.t[i] = fst.t
            
            # True head position
            true_track.x[i] = dat['center_x']
            true_track.y[i] = dat['center_y']
            
            # Estimated head position
            x,  y = self.predict_position(frame,  Nlevel,  verbose=verbose)
            est_track.x[i] = x / scale_factor
            est_track.y[i] = y / scale_factor
            
            # True head orientation
            true_track.angle[i] = dat['angle']
            
            # Estimated head orientation
            pos = {'x': x,  'y': y}
            angle,  w_angle = self.predict_orientation(frame, pos,  Nclass, verbose=verbose)
            est_track.angle[i] = w_angle
        
        error = np.recarray(shape=Nframe,  dtype=[('position',  float),
                                                                                 ('orientation',  float)])

        e = (true_track.x - est_track.x)**2 + (true_track.x - est_track.y)**2
        error['position'] = np.sqrt(e)
        z = (angles2complex(true_track.angle) - angles2complex(est_track.angle))**2
        error['orientation'] = np.abs(complex2angles(np.sqrt(z)))
        
        fst.close()

        return est_track,  true_track,  error
            
            
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
            
        
    def predict_orientation(self, frame, pos,  Nclass, verbose=True):
        """
        """
        Nave = 10
        win_sz = 40

        self.frame = frame        
        self.restore_orientation_model(win_sz,  Nclass,  verbose=verbose)
        model = self.model_orientation
            
        offsets = np.zeros((Nave, 2))
        offsets[1:, :] = np.random.randint(-10, 10, (Nave-1, 2))
        offsets += [pos['y'],  pos['x']]
        wins = []
        for i,  offset in enumerate(offsets):
            if (offset[0] > 0) and (offset[0] < frame.shape[0]) and (offset[1] > 0) and (offset[1] < frame.shape[1]):
                wins.append(get_window(frame,  win_sz,  offset).reshape((win_sz,  win_sz,  1)))
        feed_dict={model.X: np.array(wins), model.keep_prob: 1.0}
        #import pdb
        #pdb.set_trace()
        predictions = softmax(model.logits.eval(session=model.session,  feed_dict=feed_dict))
        y = np.argmax(predictions,  axis=1)
        angles = class2angle(y,  Nclass)
        w = predictions.max(axis=1)/predictions.max(axis=1).sum()
        w_z = (angles2complex(angles) * w).sum()
        w_angle = complex2angles(w_z)

        return angles[0],  w_angle        
        
        
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
    
    
    def restore_orientation_model(self,  win_sz,  Nclass,  verbose=True):
        """
        """
                
        if hasattr(self,  'model_orientation'):
            msg = ('Model %s already restored.' %
                         self.model_orientation.fname.split('/')[-1])
        else:
            
            model = create_graph_orientation(win_sz,  Nclass)
            model_fn = '%s/orientation_Nclass%d_acc*.ckpt*' % (self.model_dir,  Nclass)
            model_fn = glob(model_fn)
            model_fn.sort()
            model.fname = model_fn[-1].rstrip('.meta')

            model.session = tf.Session(graph=model.graph)
            # Restore variables from disk.
            model.saver.restore(model.session, model.fname)
            self.model_orientation = model
            msg = ('Model %s restored.' %  model.fname.split('/')[-1])
                       
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

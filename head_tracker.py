# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 15:20:09 2016

@author: hjalmar
"""
import tensorflow as tf
from ht_helper import HeSD, angle2class, FrameStepper, class2angle, whiten
from ht_helper import anglediff, get_max_gaze_line, CountdownPrinter
from ht_helper import angles2complex, complex2angles, softmax, get_error
from data_preparation import read_log_data
import numpy as np
import re
import os
from scipy.misc import imresize
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


class TrainModel:
    """
    """
    def __init__(self, Nclass=12,  data_dir=None,  model_dir=None):
    
        if data_dir is None:
            data_dir = '/home/hjalmar/head_tracker/data/CAM/BIG'
        self.data_dir = data_dir.rstrip('/')            

        if not os.path.isdir(data_dir):
            raise FileNotFoundError('data_dir %s\nis not a directory.' %
                             self.data_dir)
        if model_dir is None:
            model_dir = '/home/hjalmar/head_tracker/model/CAM/BIG'
        self.model_dir = model_dir.rstrip('/')
        
        if not os.path.isdir(model_dir):
            raise FileNotFoundError('model_dir %s\nis not a directory.' %
                             self.model_dir)
        
        self.Nclass = Nclass
        self.im_h = 120
        self.im_w = 160
        self.batch_sz = 64
            
    def get_inputs(self, fname, Nepoch, Nex_per_epoch, train=False, batch_sz=None):
        """
        Nex_per_epoch  - Ntrain or Nvalid: number_of_examples_per_epoch
        """
        if not os.path.isfile(fname):
            raise FileNotFoundError('Failed to find file: %s' % fname)
        
        if batch_sz is None:
            batch_sz = self.batch_sz

        with tf.name_scope('input'):
            fname_queue = tf.train.string_input_producer(
                [fname], num_epochs=Nepoch)

            # Even when reading in multiple threads, share the filename
            # queue.
            im, angle, angle_ok, pos_x, pos_y = self._read_and_decode(fname_queue)

            if train:
                # Distort im
                im = self._distort_inputs(im)
                n_threads = 8
            else:
                n_threads = 4
                
            # Subtract off the mean and divide by the variance of the pixels.
            im = tf.image.per_image_whitening(im)
                
            # Shuffle the examples and collect them into batch_sz batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            # Ensures a minimum amount of shuffling of examples.             
            min_queue_examples = int(Nex_per_epoch * 0.4)
            capacity = min_queue_examples + 3 * batch_sz

            im, angle, angle_ok, pos_x, pos_y = tf.train.shuffle_batch([im,
                                                                        angle,
                                                                        angle_ok,
                                                                        pos_x,
                                                                        pos_y],
                                                                        batch_size=batch_sz,
                                                                        num_threads=n_threads,
                                                                        capacity=capacity,
                                                                        min_after_dequeue=min_queue_examples)
                                                                      
            return im, angle, angle_ok, pos_x, pos_y

    
    def _read_and_decode(self,  fname_queue):
        reader = tf.TFRecordReader()
        _,  serialized_example = reader.read(fname_queue)
        features = tf.parse_single_example(
            serialized_example, 
            features={'image_raw': tf.FixedLenFeature([],  tf.string), 
                      'angle': tf.FixedLenFeature([],  tf.int64),
                      'angle_ok': tf.FixedLenFeature([],  tf.int64),
                      'position_x': tf.FixedLenFeature([],  tf.int64),
                      'position_y': tf.FixedLenFeature([],  tf.int64)})
  
        im = tf.decode_raw(features['image_raw'], tf.uint8)
        im.set_shape([self.im_h *  self.im_w])
        im = tf.reshape(im,  [self.im_h,  self.im_w,  1])
      
        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        im = tf.cast(im, tf.float32) * (1. / 255) - 0.5        
        # Convert label from a scalar uint8 tensor to an int32 scalar.
        angle = tf.cast(features['angle'], tf.int32)
        angle_ok = tf.cast(features['angle_ok'], tf.int32)
        position_x = tf.cast(features['position_x'], tf.int32)        
        position_y = tf.cast(features['position_y'], tf.int32)        
       
        return im, angle, angle_ok, position_x, position_y
            
    
    def _distort_inputs(self,  im):
        """
        Don't flip orientation images
        """ 
        im = tf.image.random_brightness(im, max_delta=63)
        im = tf.image.random_contrast(im, lower=0.2, upper=1.8)
        
        return im


    def train(self, Nepoch, lmbda=5e-4):
        """
        """
        model_fname = os.path.join(self.model_dir, 'CAM')
        train_fname = os.path.join(self.data_dir, 'train_CAM_N*.tfrecords')
        valid_fname = os.path.join(self.data_dir, 'dev_CAM_N*.tfrecords')        
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
                           
        batch_sz = self.batch_sz
                
        Nvalid = int(re.search(r'[\d]{4,6}', valid_fname.split('/')[-1]).group())
        Ntrain = int(re.search(r'[\d]{4,6}', train_fname.split('/')[-1]).group())
        Nbatch_per_epoch = Ntrain // batch_sz
        #Nbatch = Nbatch_per_epoch * Nepoch
        valid_batch_sz = 50
                
        learning_rate = 1e-4
                
        valid_X,  valid_y = [],  []
        
        model = Model(Nclass=self.Nclass, im_w=self.im_w, im_h=self.im_h, lmbda=lmbda)
        
        print('Starting training for %d epochs.' % Nepoch)
        with model.graph.as_default():
            # Input images and labels.
            images, angles, angles_ok, _, _ = self.get_inputs(train_fname,
                                                              Nepoch,
                                                              Ntrain,
                                                              train=True)
            valid_images, valid_angles, valid_angles_ok, _, _ = self.get_inputs(valid_fname, 1, Nvalid, train=False, batch_sz=valid_batch_sz)
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.loss) 
            
            with tf.Session(graph=model.graph) as session:
                                
                session.run(tf.initialize_all_variables())
                session.run(tf.initialize_local_variables())
                
                # Start input enqueue threads
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=session,  coord=coord)  

                validation_accuracy = []
                train_accuracy = [] 

                print('%s\n step |  loss |  acc  | epoch \n%s' % ('='*30, '='*30))

                step, epoch  = 0,  0

                while (epoch < Nepoch) and not coord.should_stop():
                    step += 1                        
                    # Train
                    X,  theta, theta_ok = session.run([images, angles,
                                                       angles_ok])
                                         
                    y = angle2class(theta, self.Nclass,
                                    angles_ok=theta_ok, units='deg')
                    
                    optimizer.run(feed_dict={model.X: X, model.y_: y})
                                          
                    if (step % Nbatch_per_epoch == 0):
                        l, acc = session.run([model.loss, model.accuracy],
                                             feed_dict={model.X: X,
                                                        model.y_: y})
                        epoch += 1
                        print(' %-5d| %-6.3f| %-6.2f| %-5d' % (step, l,
                                                               acc, epoch))

                        if (epoch % 10 == 0) or (epoch == Nepoch):
                            v_acc,  i = 0.0,  0
                            if len(valid_y) < 1:
                                load_valid = True
                            else:
                                load_valid = False
                                
                            while i < (Nvalid // valid_batch_sz):
                                if load_valid:
                                    X,  theta, theta_ok = session.run([valid_images,
                                                                       valid_angles,
                                                                       valid_angles_ok])
                                    y = angle2class(theta, self.Nclass,
                                                    angles_ok=theta_ok,
                                                    units='deg')
                                    valid_X.append(X)
                                    valid_y.append(y) 
                                feed_dict = {model.X: valid_X[i],
                                             model.y_: valid_y[i]}
                                v_acc += model.accuracy.eval(feed_dict=feed_dict)
                                i += 1
                            validation_accuracy.append(v_acc/i)
                            train_accuracy.append(np.mean(acc))
                            model.saver.save(session, ('%s_Nclass%d_acc%1.1f_%d.ckpt' %
                                                       (model_fname, 
                                                        self.Nclass,
                                                        validation_accuracy[-1],
                                                        epoch)))

                print('Done training for %d epochs, %d steps.' % (epoch, step-1))       
                                                     
                # Ask threads to stop
                coord.request_stop()

                # Wait for threads to finish.
                coord.join(threads)
                session.close() 
                
                print('Training accuracy:',  train_accuracy)
                print('Validation accuracy:',  validation_accuracy)
                
        return validation_accuracy, train_accuracy
              
          
class Model:
    
    def __init__(self, Nclass, im_w, im_h, lmbda=5e-4):
        
        self.Nclass = Nclass
        self.im_w = im_w
        self.im_h = im_h
    
        self.graph = tf.Graph()
        # Define ops and tensors in `g`.
        with self.graph.as_default():
            # Input data.
            self.X = tf.placeholder(tf.float32, shape=(None, im_h, im_w, 1))
            self.y_ = tf.placeholder(tf.float32, shape=(None))
            
            c1 = tf.nn.relu(self._conv_layer(self.X, (11, 11, 1, 32), "conv1"))
            c2 = tf.nn.relu(self._conv_layer(c1, (5, 5, 32, 64), "conv2"))
            p1 = tf.nn.max_pool(c2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool1')
            c3 = tf.nn.relu(self._conv_layer(p1, (3, 3, 64, 128), "conv3"))
            c4 = tf.nn.relu(self._conv_layer(c3, (3, 3, 128, 256), "conv4"))
            p2 = tf.nn.max_pool(c4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool2')
            c5 = tf.nn.relu(self._conv_layer(p2, (3, 3, 256, 256), "conv5"))
            self.top_conv = self._conv_layer(c5, (3, 3, 256, 1024), "conv6")
            gap = tf.reduce_mean(self.top_conv, [1,2])  # Global Average Pooling
            
            with tf.variable_scope("GAP"):
                shape = (1024, Nclass)
                w_init = tf.truncated_normal_initializer(mean=0.0, stddev=HeSD(shape))
                gap_w = tf.get_variable("W", shape=shape, initializer=w_init)

            self.logits = tf.matmul(gap, gap_w)
            
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        self.logits, tf.to_int64(self.y_), name='xentropy')
            self.loss = tf.reduce_mean(xentropy, name='xentropy_mean')
            
            weights = filter(lambda x: x.name.endswith('W:0'), tf.trainable_variables())
            regularizer = tf.reduce_sum(tf.pack([tf.nn.l2_loss(x) for x in weights]))
            #self.loss += (regularizer * 5e-4)
            self.loss += (regularizer * lmbda)
            
            correct = tf.equal(tf.argmax(self.logits, 1), tf.cast(self.y_,  tf.int64))
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) * 100.
            
            # CAM
            top_conv_resz = tf.image.resize_bilinear(self.top_conv,
                                                     [self.im_h, self.im_w])
            label_w = tf.gather(tf.transpose(gap_w), tf.cast(self.y_, tf.int32))
            label_w = tf.reshape(label_w, [-1, 1024, 1])
            top_conv_resz = tf.reshape(top_conv_resz,
                                       [-1, self.im_h * self.im_w, 1024])   
            cam = tf.batch_matmul(top_conv_resz, label_w)
            self.cam = tf.reshape(cam, [-1, self.im_h, self.im_w])

            self.saver = tf.train.Saver()
            

    def _conv_layer(self, z, shape, name):
    
        with tf.variable_scope(name):
            w_init = tf.truncated_normal_initializer(mean=0.0,
                                                     stddev=HeSD(shape))
            w = tf.get_variable("W", shape=shape, initializer=w_init)
            
            b = tf.get_variable("b", shape=shape[-1],
                                initializer=tf.constant_initializer(0.1))
    
            conv = tf.nn.conv2d(z, w, [1, 1, 1, 1], padding='SAME')
    
        return tf.nn.bias_add(conv, b)
                        

class HeadTracker:
    """
    """
    def __init__(self, Nclass=13, model_dir=None, im_w=160, im_h=120):
    
        if model_dir is None:
                model_dir = '/home/hjalmar/head_tracker/model/CAM'
        self.model_dir = model_dir.rstrip('/')
        
        self.Nclass = Nclass
        self.im_h = im_h
        self.im_w = im_w
        self.im_scale = self.im_w / 640.  # frame.shape is (480, 640)
        self.frame = None
        
                
    def track2video(self, in_fname, out_fname, log_fname=None,
                    t_start=0.0, t_end=-1, dur=None, verbose=True):
        """
        t_start  : only used if no log_fname is provided
        t_end  : only used if no log_fname is provided
        dur  : only used if no log_fname is provided
        """
        if not tf.gfile.Exists(in_fname):
            raise ValueError('Failed to find file: %s' % in_fname)

        fst = FrameStepper(in_fname)
        fps = int(round(1/fst.dt))
        FFMpegWriter = manimation.writers['ffmpeg']
        ttl = 'Head position tracking from video %s.'  % in_fname.split('/')[-1]
        metadata = dict(title=ttl, artist='Matplotlib',
                comment='more info...')  # TODO!
        writer = FFMpegWriter(fps=fps, metadata=metadata, bitrate=20000, codec=None)  # TODO: set a good codec
        dpi = 96
        figsize = (fst.frame.shape[1]/dpi, fst.frame.shape[0]/dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)  # TODO dpi depends on the monitor used, remove this dependence
                                                                         # see: http://stackoverflow.com/questions/13714454/specifying-and-saving-a-figure-with-exact-size-in-pixels 

        if t_start < 0:
            raise ValueError('t_start cannot be less than 0.0 (beginning of the video).')
        if t_end < 0:
            t_end = fst.duration
        if not dur is None:
            t_end = min(t_end,  t_start + dur)
        if t_end > fst.duration:
            raise ValueError('t_end cannot be later %1.3f (time of the last frame)' %
                             fst.duration)
                             
        if not log_fname is None:
            if not tf.gfile.Exists(log_fname):
                raise ValueError('Failed to find file: %s' % log_fname)
            else:
                log_data, log_header = read_log_data(log_fname)
                Nframe = len(log_data)
                
                if verbose:
                    # Counter printed on command line
                    cdp = CountdownPrinter(Nframe)
            
                with writer.saving(fig, out_fname, dpi):
                    for i, dat in enumerate(log_data):
                        
                        if verbose:
                            cdp.print(i)

                        fst.read_t(dat['frame_time'])
                        true_pos = {'x': dat['center_x'], 'y': dat['center_y']}
                        if dat['angle_ok']:
                            true_angle = (180 * (dat['angle'] / np.pi)).round()
                        else:
                            true_angle = None
                        self.plot(fst.frame, true_pos=true_pos, 
                                  true_angle=true_angle, fig=fig, verbose=False)
                        writer.grab_frame()
                        fig.clf()                        
        else:
        
            Nframe = int(np.ceil((t_end - t_start) / fst.dt))
            if verbose:
                # Counter printed on command line
                cdp = CountdownPrinter(Nframe)            
            
            with writer.saving(fig, out_fname, dpi):
                ok = fst.read_t(t_start)
                i = 0
                while (fst.t < t_end) and ok: 
                    
                    if verbose:
                        cdp.print(i)
    
                    self.plot(fst.frame, true_pos=None, fig=fig, verbose=False)
                    writer.grab_frame()
                    fig.clf()
                    ok = fst.next()
                    i += ok

        fst.close() 


    def track2fig(self, in_fname, out_fname, log_data, verbose=True):
        """
        """
        if not tf.gfile.Exists(in_fname):
            raise ValueError('Failed to find file: %s' % in_fname)
                        
        fst = FrameStepper(in_fname)
        #figsize=figsize, dpi=dpi
        fig = plt.figure()
        Nframe = len(log_data)
        if verbose:
            # Counter printed on command line
            cdp = CountdownPrinter(Nframe)            
            
        for i, dat in enumerate(log_data):

            if verbose:
                cdp.print(i)
            
            print(i, dat['frame_time'])
            fst.read_t(dat['frame_time'])
            true_pos = {'x': dat['center_x'], 'y': dat['center_y']}
            if dat['angle_ok']:
                true_angle = (180 * (dat['angle'] / np.pi)).round()
            else:
                true_angle = None

            self.plot(fst.frame, true_pos=true_pos, 
                      true_angle=true_angle, fig=fig, verbose=False)
            fig.savefig('%s_%03d.svg' % (out_fname, i))
            fig.savefig('%s_%03d.png' % (out_fname, i))
            fig.clf()

        fst.close()   
                
                
    def track(self, video_fname, t_start=0.0, t_end=-1, dur=None, verbose=True):
        """
        """
        if not tf.gfile.Exists(video_fname):
            raise ValueError('Failed to find file: %s' % video_fname)
        
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
        
        Nframe = int(np.ceil((t_end - t_start) / fst.dt))
        
        if verbose:
            cdp = CountdownPrinter(Nframe)

        est_track = np.recarray(shape=Nframe+1,
                                dtype=[('t',  float), ('x',  float),
                                       ('y',  float), ('angle',  float),
                                       ('angle_w', float)])        
        
        i = 0
        ok = fst.read_t(t_start)
        while (fst.t < t_end) and ok:
            if verbose:
                cdp.print(i)
            x,  y, angle, angle_w, _ = self.predict(fst.frame, verbose=False)
            est_track[i].x = x
            est_track[i].y = y
            est_track[i].angle = angle
            est_track[i].angle_w = angle_w            
            est_track[i].t = fst.t
            ok = fst.next()
            i += ok
            
        est_track = est_track[:i]
            
        fst.close()
            
        return est_track
        
        
    def test_track(self, log_fname, video_dir, Nframe=None):
        """
        Nframe  : number of frames to predict.
                  Default all frames in the log file.
        """
        verbose=False
        log_data, log_header = read_log_data(log_fname)

        if Nframe is None:
            Nframe = len(log_data) - 1

        if Nframe >= len(log_data):
            raise ValueError('Nframes cannot be greater than the number of frames in the log file.')
            
        #video_fname = '%s/%s' % (video_dir.rstrip('/'), log_header['video_fname'])
        video_fname = os.path.join(video_dir.rstrip('/'),
                                   log_header['video_fname'])
        video_fname = glob(video_fname)[0]
        fst = FrameStepper(video_fname)
        
        est_track = np.recarray(shape=Nframe,
                                dtype=[('t',  float), ('x',  float),
                                       ('y',  float), ('angle',  float),
                                       ('angle_w', float)])
        true_track = np.recarray(shape=Nframe,
                                 dtype=[('t',  float), ('x',  float),
                                        ('y',  float), ('angle',  float)])
                          

        if verbose:
            cdp = CountdownPrinter(Nframe)
                          
        for i, dat in enumerate(log_data[:Nframe]):
            
            if verbose:
                cdp.print(i)
            # Read the frame
            fst.read_t(dat['frame_time'])
            
            # Time of frame
            true_track[i].t = fst.t
            est_track[i].t = fst.t
            
            # True head position
            true_track[i].x = dat['center_x']
            true_track[i].y = dat['center_y']
            # True head orientation
            if not dat['angle_ok']:
                true_track[i].angle = np.nan
            else:
                true_track[i].angle = 180. * (dat['angle'] / np.pi)
            
            # Estimated head position and orientation
            x,  y, angle, angle_w, _ = self.predict(fst.frame, verbose=verbose)
            est_track[i].x = x
            est_track[i].y = y
            est_track[i].angle = angle
            est_track[i].angle_w = angle_w
                                   
        fst.close()
        
        error, error_desrc = get_error(est_track, true_track)

        return est_track, true_track, error, error_desrc
            
        
    def predict(self, frame, verbose=True):
        """
        Frame by frame
        
        x, y -- in frame coordinates
        """

        self.restore_model(verbose=verbose)

        if frame.ndim == 3:
            frame = frame.mean(axis=2)
        
        rescale = False
        if frame.shape[0] == 480 and frame.shape[1] == 640:
            im = imresize(frame, self.im_scale)
            rescale = True
        elif frame.shape[0] == self.im_h and frame.shape[1] == self.im_w:
            im = frame
        else:
            raise ValueError('Some odd differences btw frame.shape and'
                             ' self.im_w/im_w. FIX this.')
        # Reshape and whiten the image
        im = whiten(im.astype(float)).reshape((1, self.im_h, self.im_w, 1))

        p = softmax(self.model.logits.eval(session=self.model.session, 
                                           feed_dict={self.model.X: im}))
        label = p.argmax()
        angles = class2angle(np.arange(self.Nclass-1), self.Nclass-1)
        # Use the Softmax output, p, as weights for a weighted average.
        p = (p[0, :-1] / p[0, :-1].sum()).flatten()
        z_w = (angles2complex(angles) * p).sum()
        angle_w = complex2angles(z_w)
        
        if (label == (self.Nclass - 1)):   # head orientation is the horiz plane not visible.
            angle = np.nan
            angle_w = np.nan
        else:
            angle = angles[label]
            
        cam = self.model.cam.eval(session=self.model.session,
                                  feed_dict={self.model.X: im,
                                             self.model.y_: label})
        
        # rescale cam to the same size as frame
        if rescale:
            cam = imresize(cam.reshape((self.im_h, self.im_w)), 1/self.im_scale)
        else:
            cam = cam.reshape((self.im_h, self.im_w))
            
        y, x = np.unravel_index(cam.argmax(), cam.shape)
                
        return x, y, angle, angle_w, cam
        
    
    def restore_model(self, verbose=True):
        """
        """
                
        if hasattr(self,  'model'):
            msg = ('Model %s already restored.' % 
                   self.model.fname.split('/')[-1])
        else:
            
            model = Model(Nclass=self.Nclass, im_w=self.im_w, im_h=self.im_h)

            model_fn = os.path.join(self.model_dir,
                                    'CAM_Nclass%d_acc*.ckpt' % self.Nclass)
            #model_fn = '%s/CAM_Nclass%d_acc*.ckpt' % (self.model_dir, self.Nclass)
            model_fn = glob(model_fn)
            model_fn.sort()

            if model_fn[-1].endswith('meta'):
                model.fname = model_fn[-1].rstrip('.meta')
            else:
                model.fname = model_fn[-1]
            # Following rlrs's comment on:
            # https://github.com/tensorflow/tensorflow/issues/1325
            # seems to be neccesary for getting access to the GAP weights
            model_fn_meta = glob('%s.meta' % model.fname)[0]
            saved = tf.train.import_meta_graph(model_fn_meta)
            model.session = tf.Session(graph=model.graph)
            saved.restore(model.session, model.fname)
            # Restore variables from disk.
            #model.saver.restore(model.session, model.fname)
            self.model = model
            msg = ('Model %s restored.' %  model.fname.split('/')[-1])
                       
        if verbose:
            print(msg)    
    
    
    def plot(self, frame, true_pos=None, true_angle=None,
             fname=None, fig=None, verbose=False):
        """
        """
        
        x, y, angle, angle_w, cam = self.predict(frame, verbose=verbose)

        if fig is None:
            fig = plt.figure(frameon=False)
                       
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(frame)
        im_h, im_w = frame.shape[:2]
        plt.hold(True)
        ax.imshow(cam, cmap=plt.cm.jet, alpha=0.3, interpolation='bilinear')        
        if not np.isnan(angle):
            ax.plot(x, y, 'o', ms=5, mec=[1, 0.6, 0.3], mfc='none', mew=1)
            ax.plot(x, y, 'o', ms=20, mec=[1, 0.6, 0.3], mfc='none', mew=1)
            x1, y1 = get_max_gaze_line(angle, x, y, im_w, im_h, units='deg')
            ax.plot([x, x1], [y, y1],  '-', color=[1, 0.6, 0.2], lw=2, label='argmax')
            x1, y1 = get_max_gaze_line(angle_w, x, y, im_w, im_h, units='deg')
            ax.plot([x, x1], [y, y1],  '-', color=[1, 0.3, 0.0], lw=2, label='weighted')
        else:
            ax.plot(x, y, 'o', ms=20, mfc='w', mec='w', lw=2)
        
        if not true_pos is None:
            # Maximum possible error given x, y
            max_xerr, max_yerr = max(x, im_w-x), max(y, im_h-y)
            max_err = np.sqrt(max_xerr**2 + max_yerr**2)
            error = im_h * np.sqrt((x - true_pos['x'])**2 + (y - true_pos['y'])**2) / max_err
            # Note that x,y gets replaced so that true_angle will be drawn
            # starting at true_pos instead of predicted pos.
            x, y = true_pos['x'], true_pos['y']
            ax.plot(x, y, 'o', ms=5, mec='g', mfc='none', mew=1)
            ax.plot(x, y, 'o', ms=20, mec='g', mfc='none', mew=1)
            # draw position error as a bar to the right
            ax.plot([im_w-4, im_w-4], [0, error], '-', c='r', lw=4)
        if not true_angle is None:
            x1, y1 = get_max_gaze_line(true_angle, x, y, im_w, im_h, units='deg')
            ax.plot([x, x1], [y, y1],  '-', color=[.3, 1., 0.], lw=2, label='True')          
            error_w = im_h * np.abs(anglediff(true_angle, angle_w, 'deg')) / 180
            error = im_h * np.abs(anglediff(true_angle, angle, 'deg')) / 180
            # Draw orientation error as a bar to the left
            ax.plot([4, 4], [0, error], '-', c=[1, .6, .2], lw=4)
            ax.plot([11, 11], [0, error_w], '-', c=[1, .3, 0.], lw=4)

        ax.set_xlim([0, im_w])
        ax.set_ylim([0, im_h])
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.legend()
        if not fname is None:
            fig.savefig(fname)
            plt.close(fig)  


    def close(self):
        """
        """

        if hasattr(self, 'model'):
            if hasattr(self.model, 'session'):
                self.model.session.close()
       

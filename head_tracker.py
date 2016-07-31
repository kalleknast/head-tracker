# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 15:20:09 2016

@author: hjalmar
"""
import tensorflow as tf
from ht_helper import HeSD, angle2class,  FrameStepper,  class2angle
from ht_helper import angles2complex,  complex2angles,  get_gaze_line, softmax
from data_preparation import read_log_data
import numpy as np
import re
#from scipy.ndimage.measurements import center_of_mass
from scipy.misc import imresize
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as manimation

import ipdb


class TrainModel:
    """
    """
    def __init__(self, Nclass=12,  data_dir=None,  model_dir=None):
    
        if data_dir is None:
            data_dir = '/home/hjalmar/head_tracker/data/CAM/BIG'
        self.data_dir = data_dir.rstrip('/')            
        if model_dir is None:
            model_dir = '/home/hjalmar/head_tracker/model/CAM/BIG'
        self.model_dir = model_dir.rstrip('/')
        
        self.Nclass = Nclass
        self.im_h = 120
        self.im_w = 160
        self.batch_sz = 64
            
    def get_inputs(self, fname, Nepoch, Nex_per_epoch, train=False):
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
                
            # Subtract off the mean and divide by the variance of the pixels.
            image = tf.image.per_image_whitening(image)                
                
            # Shuffle the examples and collect them into batch_sz batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            # Ensures a minimum amount of shuffling of examples.             
            min_queue_examples = int(Nex_per_epoch * 0.4)
            capacity = min_queue_examples + 3 * self.batch_sz

            images, labels = tf.train.shuffle_batch([image, label],
                                                    batch_size=self.batch_sz,
                                                    num_threads=n_threads,
                                                    capacity=capacity,
                                                    min_after_dequeue=min_queue_examples)
                                                                      
            return images, labels            

    
    def _read_and_decode(self,  fname_queue):
        reader = tf.TFRecordReader()
        _,  serialized_example = reader.read(fname_queue)
        features = tf.parse_single_example(
            serialized_example, 
            features={'image_raw': tf.FixedLenFeature([],  tf.string), 
                      'label': tf.FixedLenFeature([],  tf.int64)})
  
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([self.im_h *  self.im_w])
        image = tf.reshape(image,  [self.im_h,  self.im_w,  1])
      
        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5        
        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['label'], tf.int32)

        return image, label
            
    
    def _distort_inputs(self,  image):
        """
        Don't flip orientation images
        """ 
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        
        return image


    def train(self, Nepoch):
        """
        """
        model_fname = '%s/CAM' % (self.model_dir)
        train_fname = '%s/train_CAM_N*.tfrecords' % (self.data_dir)
        valid_fname = '%s/dev_CAM_N*.tfrecords' % (self.data_dir)
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
                
        Nvalid = int(re.search(r'[\d]{4,6}', valid_fname).group())
        Ntrain = int(re.search(r'[\d]{4,6}', train_fname).group())
        Nbatch_per_epoch = Ntrain // batch_sz
        #Nbatch = Nbatch_per_epoch * Nepoch
                
        Nclass = self.Nclass
        learning_rate = 1e-4
                
        valid_X,  valid_y = [],  []
        
        model = Model(Nclass=Nclass, im_w=self.im_w, im_h=self.im_h)
                
        with model.graph.as_default():
            # Input images and labels.
            images, labels = self.get_inputs(train_fname,  Nepoch,  Ntrain,  train=True)
            valid_images, valid_labels = self.get_inputs(valid_fname, 1,  Nvalid, train=False)

            with tf.Session(graph=model.graph) as session:
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.loss) 
                session.run(tf.initialize_all_variables())
                # Start input enqueue threads
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=session,  coord=coord)  
      
                validation_accuracy = []
                train_accuracy = [] 
                

                print('%s\n step |  loss |  acc  | epoch \n%s' % ('='*30, '='*30))

                try:
                    step, epoch  = 0,  0

                    while (epoch < Nepoch) and not coord.should_stop():
                        step += 1                        
                        # Train
                        X,  angles = session.run([images,  labels])  
                        y = angle2class(angles,  Nclass)
                        
                        optimizer.run(feed_dict={model.X: X, model.y_: y})
                                              
                        if (step % Nbatch_per_epoch == 0):
                            l, acc = session.run([model.loss, model.accuracy],
                                                 feed_dict={model.X: X, model.y_: y})                                                        
                            epoch += 1
                            print(' %-5d| %-6.3f| %-6.2f| %-5d'  % (step, l, acc, epoch))

                            if (epoch % 5 == 0):
                                v_acc,  i = 0.0,  0
                                if len(valid_y) < 1:
                                    load_valid = True
                                else:
                                    load_valid = False
                                while i < Nvalid // self.batch_sz:
                                    if load_valid and not coord.should_stop():
                                        X,  angles = session.run([valid_images,  valid_labels]) 
                                        valid_X.append(X)
                                        valid_y.append(angle2class(angles,  Nclass)) 
                                    feed_dict = {model.X: valid_X[i], model.y_: valid_y[i]}
                                    v_acc += model.accuracy.eval(feed_dict=feed_dict)
                                    i += 1
                                validation_accuracy.append(v_acc/i)
                                train_accuracy.append(np.mean(acc))
                                model.saver.save(session, 
                                                 '%s_Nclass%d_acc%1.1f_%d.ckpt' % 
                                                 (model_fname, Nclass, validation_accuracy[-1],  epoch))                                

                except tf.errors.OutOfRangeError:
                    print('Done training for %d epochs, %d steps.' % (epoch, step-1))       
                                            
                finally:                   
                    v_acc,  i = 0.0,  0
                    while i < Nvalid // self.batch_sz:
                        if load_valid and not coord.should_stop():
                            X,  y = session.run([valid_images,  valid_labels]) 
                            valid_X.append(X)
                            valid_y.append(angle2class(y, Nclass))
                        feed_dict = {model.X: valid_X[i], model.y_: valid_y[i]}
                        v_acc += model.accuracy.eval(feed_dict=feed_dict)
                        i += 1
                    validation_accuracy.append(v_acc/i)
                    model.saver.save(session, 
                                     '%s_Nclass%d_acc%1.1f_%d.ckpt' % 
                                     (model_fname, Nclass, validation_accuracy[-1],  epoch))
                
                    # Ask threads to stop
                    coord.request_stop()

                # Wait for threads to finish.
                coord.join(threads)
                session.close() 
                
                print('Training accuracy:',  train_accuracy)
                print('Validation accuracy:',  validation_accuracy)
                
        return validation_accuracy[-1]
              
          
class Model:
    
    def __init__(self, Nclass, im_w=102, im_h=76):
        
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
            self.loss += regularizer * 5e-4
            
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
                        

#    def get_CAM(self, label):
#        """
#        Return the Class Activation Map
#        """
#        top_conv_resized = tf.image.resize_bilinear(self.top_conv,
#                                                    [self.im_h, self.im_w]) # TODO: is this the upsampling?
#        # ie "By simply upsampling the class activation map to the size of the input image,
#        # we can identify the image regions most relevant to the particular category."
#        # wouldn't it be better to upsample after multiplication?
#        
#        # Some bug rules out this option, see:
#        # https://github.com/tensorflow/tensorflow/issues/1325
#        #with tf.variable_scope("GAP", reuse=True):
#        #    label_w = tf.gather(tf.transpose(tf.get_variable("W")), label)
#        
#        #  the hack
#        for w in tf.trainable_variables():
#            if w.name == 'GAP/W:0':
#                label_w = w
#                break
#
#        label_w = tf.gather(tf.transpose(label_w), label)
#        label_w = tf.reshape(label_w, [-1, 1024, 1])
#        top_conv_resized = tf.reshape(top_conv_resized,
#                                      [-1, self.im_h * self.im_w, 1024])
#        try:
#            cam = tf.batch_matmul(top_conv_resized, label_w)
#        except:
#            ipdb.set_trace()
#        cam = tf.reshape(cam, [-1, self.im_h, self.im_w])
#        
#        return cam


class HeadTracker:
    """
    """
    def __init__(self, Nclass=12, model_dir=None, im_w=160, im_h=120):
    
        if model_dir is None:
                model_dir = '/home/hjalmar/head_tracker/model/CAM'
        self.model_dir = model_dir.rstrip('/')
        
        self.Nclass = Nclass
        self.im_h = im_h
        self.im_w = im_w
        self.im_scale = self.im_w / 640.  # frame.shape is (480, 640)
        self.frame = None
        
        
#    def track2video(self,  in_fname,  out_fname,  track):
#        """
#        """
#        if not tf.gfile.Exists(in_fname):
#            raise ValueError('Failed to find file: %s' % in_fname)
#            
#        fst = FrameStepper(in_fname)        
#        fps = int(round(1/fst.dt))
#        FFMpegWriter = manimation.writers['ffmpeg']
#        ttl = 'Head position tracking from video %s.'  % in_fname.split('/')[-1]
#        metadata = dict(title=ttl, artist='Matplotlib',
#                comment='more info...')  # TODO!
#        writer = FFMpegWriter(fps=fps, metadata=metadata,  codec=None)  # TODO: set a good codec
#        dpi = 96
#        figsize = (fst.frame.shape[1]/dpi, fst.frame.shape[0]/dpi)
#        fig = plt.figure(figsize=figsize, dpi=dpi)  # TODO dpi depends on the monitor used, remove this dependence
#                                                                         # see: http://stackoverflow.com/questions/13714454/specifying-and-saving-a-figure-with-exact-size-in-pixels 
#        with writer.saving(fig, out_fname, dpi):
#            for trk in track:
#                fst.read_t(trk.t)
#                
#                ax = fig.add_subplot(111)
#                ax.imshow(fst.frame,  origin='lower')
#                x,  y = trk.x,  trk.y
#                ax.plot(x,  y,  'xr',  ms=20,  mew=2)
#                gzl_x, gzl_y = get_gaze_line(trk.angle, x, y, 50, units='deg')
#                ax.plot(gzl_x,  gzl_y,  '-g',  lw=2)
#                ax.set_xlim([0,  fst.frame.shape[1]])
#                ax.set_ylim([0,  fst.frame.shape[0]]) 
#                ax.set_xticks([])
#                ax.set_yticks([])
#                ax.set_position([0,  0,  1,  1])
#                writer.grab_frame()
#                fig.clf()
#
#        fst.close()

        
    def track2video(self, in_fname, out_fname, t_start=0.0, t_end=-1, dur=None):
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

        if t_start < 0:
            raise ValueError('t_start cannot be less than 0.0 (beginning of the video).')
        if t_end < 0:
            t_end = fst.duration
        if not dur is None:
            t_end = min(t_end,  t_start + dur)
        if t_end > fst.duration:
            raise ValueError('t_end cannot be later %1.3f (time of the last frame)' %
                             fst.duration)
        
        #Nframe = int(((t_end - t_start) / fst.dt))
        
        with writer.saving(fig, out_fname, dpi):
            fst.read_t(t_start)
            while fst.t < t_end:          
                self.plot(fst.frame, true_pos = None, fig=fig, verbose=False)
                writer.grab_frame()
                fig.clf()
                try:
                    fst.next()
                except:
                    ipdb.set_trace()

        fst.close()        
                
                
    def track_head(self, video_fname, t_start=0.0, t_end=-1, dur=None):
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
        
        Nframe = int(((t_end - t_start) / fst.dt))

        est_track = np.recarray(shape=Nframe,
                                dtype=[('t',  float), ('x',  float),
                                       ('y',  float), ('angle',  float),
                                       ('angle_w', float)])        

        if Nframe > fst.tot_n:
            raise ValueError('Nframes cannot be greater than the number of frames video.')
        
        i = 0
        fst.read_t(t_start)
        while fst.t < t_end:
            x,  y, angle, angle_w, _ = self.predict(fst.frame, verbose=False)
            est_track[i].x = x
            est_track[i].y = y
            est_track[i].angle = angle
            est_track[i].angle_w = angle_w            
            est_track[i].t = fst.t
            i += 1
            try:
                fst.next()
            except:
                ipdb.set_trace()
            
        fst.close()
            
        return est_track
        
        
    def test_track_head(self,  log_fname,  video_dir,  Nframe=300):
        """
        """
        verbose=False
        Nclass = 12
        log_data, log_header = read_log_data(log_fname)
        video_fname = '%s/%s' % (video_dir.rstrip('/'), log_header['video_fname'])
        video_fname = glob(video_fname)[0]
        fst = FrameStepper(video_fname)
        
        est_track = np.recarray(shape=Nframe,
                                dtype=[('t',  float), ('x',  float),
                                       ('y',  float), ('angle',  float),
                                       ('angle_w', float)])
        true_track = np.recarray(shape=Nframe,
                                 dtype=[('t',  float), ('x',  float),
                                        ('y',  float), ('angle',  float)])
                  
        if Nframe >= len(log_data):
            raise ValueError('Nframes cannot be greater than the number of frames in the log file.')
        
        for i, dat in enumerate(log_data[:Nframe]):

            print(Nframe-i)
            # Read the frame
            fst.read_t(dat['frame_time'])
            
            # Time of frame
            true_track.t[i] = fst.t
            est_track.t[i] = fst.t
            
            # True head position
            true_track.x[i] = dat['center_x']
            true_track.y[i] = dat['center_y']
            # True head orientation
            true_track.angle[i] = dat['angle']
            
            # Estimated head position and orientation
            x,  y, angle, angle_w, _ = self.predict(fst.frame, verbose=verbose)
            est_track.x[i] = x
            est_track.y[i] = y
            est_track.angle[i] = angle
            est_track.angle_w[i] = angle_w
            
           
        error = np.recarray(shape=Nframe,  dtype=[('position', float),
                                                  ('orientation', float)])

        e = (true_track.x - est_track.x)**2 + (true_track.x - est_track.y)**2
        error['position'] = np.sqrt(e)
        z = (angles2complex(true_track.angle) - angles2complex(est_track.angle))**2
        error['orientation'] = np.abs(complex2angles(np.sqrt(z)))
        
        fst.close()

        return est_track,  true_track,  error            
            
        
    def predict(self, frame, verbose=True):
        """
        """

        im = imresize(frame.mean(axis=2), self.im_scale)
        im = im.reshape((1, self.im_h, self.im_w, 1))
        self.restore_model(self.Nclass,  verbose=verbose)
            
        p = softmax(self.model.logits.eval(session=self.model.session, 
                                           feed_dict={self.model.X: im}))
        angles = class2angle(np.arange(self.Nclass), self.Nclass)
        # Use theSoftmax output, p, as weights for a weighted average.
        w_z = (angles2complex(angles) * p).sum()
        w_angle = complex2angles(w_z)
        label = p.argmax()
        angle = angles[label]
        
        cam = self.model.cam.eval(session=self.model.session,
                                  feed_dict={self.model.X: im,
                                             self.model.y_: label})
        
        cam = imresize(cam.reshape((self.im_h, self.im_w)), 1/self.im_scale)
        y, x = np.unravel_index(cam.argmax(), cam.shape)
        
        return x, y, angle,  w_angle, cam
        
    
    def restore_model(self,  Nclass,  verbose=True):
        """
        """
                
        if hasattr(self,  'model'):
            msg = ('Model %s already restored.' % 
                   self.model.fname.split('/')[-1])
        else:
            
            model = Model(Nclass=Nclass, im_w=self.im_w, im_h=self.im_h)

            model_fn = '%s/CAM_Nclass%d_acc*.ckpt' % (self.model_dir,  Nclass)
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
    
    
    def plot(self, frame, true_pos = None, fname=None, fig=None, verbose=False):
        
        x, y, angle, w_angle, cam = self.predict(frame, verbose=verbose)

        if fig is None:
            fig = plt.figure(frameon=False)
            
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(frame)
        plt.hold(True)
        ax.imshow(cam, cmap=plt.cm.jet, alpha=0.3, interpolation='bilinear')
        ax.plot(x, y, 'ow', ms=16, mfc='none', lw=2)
        gzl_x, gzl_y = get_gaze_line(angle, x, y, 50, units='deg')
        ax.plot(gzl_x,  gzl_y,  '-w',  lw=2, label='argmax')
        gzl_x, gzl_y = get_gaze_line(w_angle, x, y, 50, units='deg')
        ax.plot(gzl_x,  gzl_y,  '-r',  lw=2, label='weighted')
        ax.set_xlim([0, 640])
        ax.set_ylim([0, 480])
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.legend()
        if not fname is None:
            fig.savefig(fname)
            plt.close(fig)  


    def close(self):
        """
        """

        self.model.session.close()

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:33:25 2016

@author: hjalmar
"""


from ht_helper import get_input, FrameStepper, CountdownPrinter
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.misc import imresize
from time import sleep
from glob import glob
import numpy as np
import warnings
import tensorflow as tf

warnings.simplefilter("ignore")


class LabelFrames:
    """
    """

    def __init__(self, video_fname, log_fname=None):
        """
        """
        # Supress the plt.ginput warning.
        warnings.warn("deprecated", DeprecationWarning)
        plt.ion()

        self.start_t = datetime.now()
        self.video_fname = video_fname

        if log_fname is None:

            self.log_fname = ('%s.txt' %
                              video_fname.split('/')[-1].split('.')[-2])
        else:
            self.log_fname = log_fname

        # Open output log file.
        self.f = open(self.log_fname, 'w')
        self.figsize = [8.125, 6.125]
        self.fst = FrameStepper(self.video_fname)
        self.available_nf = self.fst.tot_n
        self.frame = self.fst.frame
        self.frame_num = self.fst.n
        self.frame_time = self.fst.t

        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_axes([0.01, 0.01, 0.97, 0.97])
        self.num_labelled = 0
        self.imwdt = self.frame.shape[1]
        self.imhgt = self.frame.shape[0]
        self.head_data = {'center_x': 0.0,
                          'center_y': 0.0,
                          'angle': 0.0,
                          'angle_visible': 0,
                          'box_width': 0.0,
                          'box_height': 0.0,
                          'forehead_pos_x': 0.0,
                          'forehead_pos_y': 0.0,
                          'tuft_pos_left_x': 0.0,
                          'tuft_pos_left_y': 0.0,
                          'tuft_pos_right_x': 0.0,
                          'tuft_pos_right_y': 0.0}

        self._head_len_wdt_ratio = 1.0  # head a bit wider than long
        self._head_backwrd_shift = 0.6
        self._head_scaling = 3  # multiplied with inter_tuft_distance to get
                                # width of head including tufts
        # Write the header to the log file.
        self._write_log(write_header=True)
        

    def run_spacedbatch(self, t0=0.2, batch_size=1000):
        """
        """
        self.batch_size = batch_size
        n_skip = self.fst.tot_n // batch_size
        
        frame_num = int(t0 / self.fst.dt)
        while frame_num <= self.fst.tot_n:
            self.fst.read_frame(frame_num)
            self.frame = self.fst.frame
            self.frame_num = self.fst.n
            self.frame_time = self.fst.t            
            self._annote(1, batch_type='spaced')
            
            if self.head_position_ok:
                self._write_log(write_header=False)
                self.num_labelled += 1  

            frame_num = np.random.randint(3, 2*n_skip-1) + self.frame_num

        print('Batch complete.\n%d frames labelled.' % self.num_labelled)            

        self.close()
        

    def run_sequentialbatch(self, t0, seq_len=200, n_seqs=10):
        """
        """
        self.max_skip = self.fst.duration - seq_len * (n_seqs - 1) * self.fst.dt - t0
        self.max_skip /= (n_seqs - 1)
        self.available_nf - seq_len * n_seqs

        for i in range(n_seqs):
            print('Starting sequence %d of %d at time %1.2fs.' %
                  (i+1, n_seqs, t0))
            self.run_sequence(t0, seq_len=seq_len)
            if i < n_seqs - 1:
                if not get_input('Continue', bool, default=True):
                    print('Quitting without having finished the batch.')
                    break
                # Start time for next sequence.
                # Random drawn from the interval [1/fps, max_skip+1/fps) s.
                t0 = self.frame_time + np.random.random() * self.max_skip + self.fst.dt
            else:
                print('Batch complete.\n%d frames labelled.' %
                      self.num_labelled)

        self.close()

    def run_sequence(self, t0, seq_len=200):
        """
        """

        self._seq_len = seq_len
        self.fst.read_t(t0)
        self.frame = self.fst.frame
        self.frame_num = self.fst.n
        self.frame_num_start = self.fst.n
        self.frame_num_end = self.fst.n + seq_len
        self.frame_time = self.fst.t

        self._annote(self.frame_num-self.frame_num_start+1)
        self._write_log(write_header=False)

        for i in range(1, seq_len + 1):
            self.fst.next()
            self.frame = self.fst.frame
            self.frame_num = self.fst.n
            self.frame_time = self.fst.t
            self._annote(self.frame_num-self.frame_num_start + 1, batch_type='sequential')
            if self.head_position_ok:
                self._write_log(write_header=False)
                self.num_labelled += 1

    def _write_log(self, write_header=False):
        """
        Writes a header or head data.
        """
        if write_header:
            s = ('date_time, video_fname, head_backwrd_shift, '
                 'head_len_wdt_ratio, head_scaling\n'
                 '%s, %s, %1.2f, %1.2f, %1.2f\n'
                 'frame_number, frame_time(s), center_x(px), center_y(px), '
                 'gaze_angle(rad), gaze_angle_visible(bool), box_width(px), box_height(px), '
                 'forehead_pos_x(px), forehead_pos_y(px), '
                 'tuft_pos_left_x(px), tuft_pos_left_y(px), '
                 'tuft_pos_right_x(px), tuft_pos_right_y(px)\n' %
                 (self.start_t.strftime("%Y-%m-%d %H:%M:%S"),
                  self.video_fname.split('/')[-1],
                  self._head_backwrd_shift,
                  self._head_len_wdt_ratio,
                  self._head_scaling))
        else:
            s = ('%d, %1.9f, %1.2f, %1.2f, %1.9f, %d, %1.2f, %1.2f, '
                 '%1.2f, %1.2f, %1.2f, %1.2f, %1.2f, %1.2f\n' %
                 (self.frame_num,
                  self.frame_time,
                  self.head_data['center_x'],
                  self.head_data['center_y'],
                  self.head_data['angle'],
                  self.head_data['angle_visible'],
                  self.head_data['box_width'],
                  self.head_data['box_height'],
                  self.head_data['forehead_pos_x'],
                  self.head_data['forehead_pos_y'],
                  self.head_data['tuft_pos_left_x'],
                  self.head_data['tuft_pos_left_y'],
                  self.head_data['tuft_pos_right_x'],
                  self.head_data['tuft_pos_right_y']))

        self.f.write(s)


    def _annote(self, frame_num_in_seq, batch_type='sequential'):
        """
        """
        redo = True
        while redo:
            # plot
            # Set image to be oriented with 0,0 in lower left corner
            # and x_max, y_max in upper right corner
            self.ax.cla()
            self.ax.imshow(self.frame, origin='lower')
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            
            if batch_type == 'sequential':
                s = ('start: %d, end: %d, current: %d, remaining: %d' %
                     (self.frame_num_start, self.frame_num_end,
                      self.frame_num_start + frame_num_in_seq,
                      self._seq_len - frame_num_in_seq))
            elif batch_type == 'spaced':
                s = ('time: %1.2fs, current: %d, remaining: ≈ %d' %
                     (self.frame_time, self.num_labelled,
                      self.batch_size - self.num_labelled))
            else:
                raise ValueError('Unknown batch_type specified: %s' % batch_type)

            txt0 = self.ax.text(self.imwdt-5, 5, s, va='bottom',
                                ha='right', fontsize=12, color=[0.4, 0.8, 0.2])

            s = 'LEFT to SELECT 1) forehead, 2) one tuft, 3) other tuft\n'
            s += 'MIDDLE to SKIP'
            txt1 = self.ax.text(self.imwdt//2, self.imhgt-10, s, va='top',
                                color='w', fontsize=14, ha='center')
            plt.draw()
            self._get_head_pos_and_rotation()
            if self.head_position_ok:
                draw_head_position_and_angle(self.ax,
                                             self.imwdt,
                                             self.imhgt,
                                             self._head_backwrd_shift,
                                             self.head_data)
            txt1.set_visible(False)
            
            s = 'Gaze direction visible?\nLEFT for Yes | MIDDLE for No'
            txt2 = self.ax.text(self.imwdt//2, self.imhgt-10, s, va='top',
                                color='w', fontsize=14, ha='center')
            plt.draw()
            r = plt.ginput(n=1, mouse_add=1, mouse_stop=2,
                           mouse_pop=3, timeout=0)

            if len(r) > 0:
                self.head_data['angle_visible'] = 1
            else:
                self.head_data['angle_visible'] = 0
                         
            txt2.set_visible(False)
            s = 'MIDDLE to REDO | LEFT to NEXT'
            self.ax.text(self.imwdt//2, self.imhgt-10, s, va='top',
                         color='w', fontsize=14, ha='center')
            plt.draw()
            r = plt.ginput(n=1, mouse_add=1, mouse_stop=2,
                           mouse_pop=3, timeout=0)

            if len(r) == 0:
                redo = True
                self.ax.cla()
            else:
                redo = False

        txt0.set_visible(False)


    def _get_head_pos_and_rotation(self):
        """
        """
        # Get tuft and forehead positions from image.
        pos = plt.ginput(3, timeout=0)

        if len(pos) == 3:
            self.head_position_ok = True
            # Get the head rotation angle.
            head_pos_x = pos[1][0] + (pos[2][0] - pos[1][0])/2
            head_pos_y = pos[1][1] + (pos[2][1] - pos[1][1])/2
            xdist = pos[0][0] - head_pos_x
            ydist = pos[0][1] - head_pos_y  # y-dim 0 is at top of image
            # Angle from horizontal
            #angle = np.angle(xdist+ydist*1j)
            angle = np.arctan2(ydist, xdist)

            # Positions of tufts relative to gaze/nose.
            tuft_pos_left_x, tuft_pos_left_y = pos[1][0], pos[1][1]
            tuft_pos_right_x, tuft_pos_right_y = pos[2][0], pos[2][1]
            # Make sure that left tuft is left relative to gaze.
            flip = False
            if (angle == 0 or angle == 2*np.pi):
                if tuft_pos_left_y < tuft_pos_right_y:
                    flip = True
            elif angle < np.pi:
                if tuft_pos_left_x > tuft_pos_right_x:
                    flip = True
            elif angle == np.pi:
                if tuft_pos_left_y > tuft_pos_right_y:
                    flip = True
            else:  # angle > pi
                if tuft_pos_left_x < tuft_pos_right_x:
                    flip = True
            if flip:
                    tmp_x, tmp_y = tuft_pos_right_x, tuft_pos_right_y
                    tuft_pos_right_x = tuft_pos_left_x
                    tuft_pos_right_y = tuft_pos_left_y
                    tuft_pos_left_x, tuft_pos_left_y = tmp_x, tmp_y

            # Store head position and angle data in head_data.
            # To be saved in the log file later.
            self.head_data['center_x'] = head_pos_x
            self.head_data['center_y'] = head_pos_y
            self.head_data['angle'] = angle
            inter_tuft_dist = np.sqrt((pos[1][0] - pos[2][0])**2 +
                                      (pos[1][1] - pos[2][1])**2)
            self.head_data['box_width'] = self._head_scaling * inter_tuft_dist
            self.head_data['box_height'] = self.head_data['box_width'] * \
                self._head_len_wdt_ratio
            self.head_data['forehead_pos_x'] = pos[0][0]
            self.head_data['forehead_pos_y'] = pos[0][1]
            self.head_data['tuft_pos_left_x'] = tuft_pos_left_x
            self.head_data['tuft_pos_left_y'] = tuft_pos_left_y
            self.head_data['tuft_pos_right_x'] = tuft_pos_right_x
            self.head_data['tuft_pos_right_y'] = tuft_pos_right_y
        else:
            self.head_position_ok = False

    def close(self):
        """
        
        """
        self.fst.vr.close()
        self.f.close()


def draw_head_position_and_angle(ax, imwdt, imhgt, head_backwrd_shift, data):
    """
    """
    # Forehead
    x, y = data['forehead_pos_x'], data['forehead_pos_y']
    ax.plot(x, y, 'o', ms=5, mec=[1, 0.6, 0.3], mfc='none', mew=1)
    ax.plot(x, y, 'o', ms=20, mec=[1, 0.9, 0.5], mfc='none', mew=1)
    # Left tuft
    x, y = data['tuft_pos_left_x'], data['tuft_pos_left_y']
    ax.plot(x, y, 'o', ms=20, mec=[1, 0.3, 0.1], mfc='none', mew=1)
    ax.plot(x, y, '.r')
    # Right tuft
    x, y = data['tuft_pos_right_x'], data['tuft_pos_right_y']
    ax.plot(x, y, 'o', ms=20, mec=[1, 0.3, 0.1], mfc='none', mew=1)
    ax.plot(x, y, '.r')
    # Head postion, center between the tufts.
    hp_x, hp_y = data['center_x'], data['center_y']
    ax.plot(hp_x, hp_y, 'o', ms=5, mec=[1, 0, 0], mfc='none', mew=1)
    ax.plot(hp_x, hp_y, 'o', ms=20, mec=[1, 0.3, 0.1], mfc='none', mew=1)

    # For plotting line of gaze.
    angle = data['angle']
    if angle < 0:               # 3rd or 4th quadrant
        ydist_lim = - hp_y + 10
    else:                       # 1st or 2nd quadrant
        ydist_lim = imhgt - hp_y - 10
    if abs(angle) < np.pi/2:    # 1st or 4th quadrant
        xdist_lim = imwdt - hp_x - 10
    else:                       # 2nd or 3rd quadrant
        xdist_lim = - hp_x + 10
    # y-dim 0 is at top of image
    xdist, ydist = data['forehead_pos_x'] - hp_x, data['forehead_pos_y'] - hp_y
    # TODO: try to replace with get_gaze_line
    # or at least check that (ydist_lim / ydist) * ydist really is neccessary
    gaze_pos_x = hp_x + (ydist_lim / ydist) * ydist / np.tan(angle)
    gaze_pos_y = hp_y + (xdist_lim / xdist) * xdist * np.tan(angle)
    # Fixing gaze line so that it doesn't end outside of image.
    if gaze_pos_y > imhgt:
        gaze_pos_y = hp_y + ydist_lim
        gaze_pos_x = hp_x + ydist_lim / np.tan(angle)
    elif gaze_pos_x > imwdt:
        gaze_pos_x = hp_x + xdist_lim
        gaze_pos_y = hp_y + xdist_lim * np.tan(angle)
    elif gaze_pos_y < 0:
        gaze_pos_y = hp_y + ydist_lim
        gaze_pos_x = hp_x + ydist_lim / np.tan(angle)
    elif gaze_pos_x < 0:
        gaze_pos_x = hp_x + xdist_lim
        gaze_pos_y = hp_y + xdist_lim * np.tan(angle)
    # Plot gaze line
    ax.plot([hp_x, gaze_pos_x], [hp_y, gaze_pos_y],
            '-', color=[1, 0.6, 0.2], lw=3, alpha=0.5)
            
    ax.set_xlim([0, imwdt])
    ax.set_ylim([0, imhgt])


def read_log_data(log_fname):
    """
    """
    dtype = [('frame_num', int),
             ('frame_time', np.float64),
             ('center_x', np.float64),
             ('center_y', np.float64),
             ('angle', np.float64),
             ('angle_ok', int),
             ('box_width', np.float64),
             ('box_height', np.float64),
             ('forehead_pos_x', np.float64),
             ('forehead_pos_y', np.float64),
             ('tuft_pos_left_x', np.float64),
             ('tuft_pos_left_y', np.float64),
             ('tuft_pos_right_x', np.float64),
             ('tuft_pos_right_y', np.float64)]

    data = np.genfromtxt(log_fname, dtype=dtype, delimiter=',', skip_header=3)
    
    f = open(log_fname, 'r')
    H0 = f.readline().split(',')
    H1 = f.readline().split(',')
    header = {}
    for h0, h1 in zip(H0, H1):
        header[h0.strip().rstrip('\n')] = h1.strip().rstrip('\n')
    f.close()
    return data, header


def plot_head_data(log_fname, video_fname):
    """
    Mainly for checking the manual annotation.
    """
    head_backwrd_shift = 0.6
    figsize = [8.125, 6.125]
    fst = FrameStepper(video_fname)
    frame = fst.frame

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.01, 0.01, 0.97, 0.97])
    imwdt = frame.shape[1]
    imhgt = frame.shape[0]

    log_data, log_header = read_log_data(log_fname)

    for i, dat in enumerate(log_data):

        fst.read_t(dat['frame_time'])
        ax.cla()
        ax.imshow(fst.frame, origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        draw_head_position_and_angle(ax, imwdt, imhgt, head_backwrd_shift, dat)
        plt.draw()
        fig.savefig('test_%d.png' % i)
        sleep(5)

    fst.close()
 
     
def convert_to_tfrecords(images, angles, angles_ok, positions, fname):   
    """
    """
    
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) 
  
    if images.shape[0] != angles.shape[0]:
        raise ValueError("Images size %d does not match label size %d." %
                         (images.shape[0], angles.shape[0]))

    writer = tf.python_io.TFRecordWriter(fname)
  
    for i in range(images.shape[0]):
        
        image_raw = images[i].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(images.shape[1]),
                'width': _int64_feature(images.shape[2]),
                'depth': _int64_feature(images.shape[3]),
                'angle': _int64_feature(int(angles[i])),
                'angle_ok': _int64_feature(int(angles_ok[i])),
                'position_x': _int64_feature(int(positions[i]['x'])),
                'position_y': _int64_feature(int(positions[i]['y'])),
                'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
        
    writer.close()    
    

#def to_record(image, label, angle, position_x, position_y):
#    """
#    Image and label to record
#
#    From:
#    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py
#    """
#    
#    def _int64_feature(value):
#        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#    
#    def _bytes_feature(value):
#        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) 
#    
#    # From whitened float image to uint8
#    image -= image.min()
#    image /= image.max()
#    image *= 255   
#    image = np.uint8(image)    
#    
#    n_row = image.shape[0]
#    n_col = image.shape[1]
#    image = image.reshape([n_row,  n_col,  1])    
#    depth = image.shape[2]    
#        
#    example = tf.train.Example(features=tf.train.Features(feature={
#            'height': _int64_feature(n_row),
#            'width': _int64_feature(n_col),
#            'depth': _int64_feature(depth),
#            'label': _int64_feature(int(label)),
#            'angle': _int64_feature(int(angle)),
#            'position_x': _int64_feature(int(position_x)),
#            'position_y': _int64_feature(int(position_y)),
#            'image_raw': _bytes_feature(image.tostring())}))
#    
#    return example.SerializeToString()


def labeledData2storage_CAM(log_dir, video_dir, data_dir, Ntrain=None, Ndev=3000):
    """
    Trying to maximize resolution while minimizing the number of windows/patches
    to classify per video fram using a Multi Resolution Pyramid.
    
    Parameters
    ----------
    log_dir
    video_dir
    data_dir
    Ntrain
    Ndev -- 3 000 should be fine for 1% acc differences.
    """
    
    #head_size = 200  # pixels in 480 x 640 frame
    #scale_factor = 32 / head_size  # TODO improve this 
    scale_factor = 160/640.
    log_fnames = glob(log_dir.rstrip('/') + '/HeadData*.txt')
    Nlogs = len(log_fnames)    
    
    frames = []
    angles_ok = []
    angles = []
    positions_x = []
    positions_y = []
   
    for i,  log_fname in enumerate(log_fnames):
        print('Processing file (%d of %d): %s' % (i+1,  Nlogs,  log_fname.split('/')[-1]))

        log_data, log_header = read_log_data(log_fname)
        video_fname = '%s/%s' % (video_dir.rstrip('/'), log_header['video_fname'])
        video_fname = glob(video_fname)
        if len(video_fname) != 1:
            print('The video file matching to the log cannot be found in %s'
                  % video_dir)
            print('Log file:', log_fname)
            print('Found video files:', video_fname)
            return 0

        # To read the frames
        fst = FrameStepper(video_fname[0])
    
        # Scale head positions to fit rescaled frames.
        log_data['center_x'] = log_data['center_x'] * scale_factor
        log_data['center_y'] = log_data['center_y'] * scale_factor
        # Head angles to degrees (-180 to 180)
        log_data['angle'] = (180 * (log_data['angle'] / np.pi)).round()
        
        # Counter printed on command line
        cdp = CountdownPrinter(log_data.shape[0])

        for j, dat in enumerate(log_data):
            # Counter printed on the command line
            cdp.print(j)        
            # Read the frame
            fst.read_t(dat['frame_time'])
            frame = imresize(fst.frame.mean(axis=2), scale_factor)
            frames.append(frame.reshape(frame.shape[0], frame.shape[1], 1))
            angles_ok.append(dat['angle_ok'])
            angles.append(dat['angle'])
            positions_x.append(dat['center_x'])
            positions_y.append(dat['center_y'])

    fst.close()
    # assign to dev and train set, and write
    print('\nSaving labeled data to tfrecords...')

    Ntot = len(frames)
    #im_h, im_w = frames[0].shape
    if Ndev is None and Ntrain is None:
        Ndev = 3000
        Ntrain = Ntot - Ndev
    elif Ndev is None and np.isscalar(Ntrain):
        Ndev = Ntot - Ntrain
    elif Ntrain is None and np.isscalar(Ndev):
        Ntrain = Ntot - Ndev
    elif np.isscalar(Ndev) and np.isscalar(Ntrain):
        Ndev = Ntot - Ntrain
        print('Ndev set to :', Ndev)
    else:
        import pdb
        pdb.set_trace()
        raise ValueError('What do you want?')

    rnd_idx = np.random.permutation(Ntot)
    dev_idx = rnd_idx[:Ndev]
    train_idx = rnd_idx[Ndev:]
    train_idx = train_idx[:Ntrain]
    
    images = np.array(frames, dtype=np.uint8)
    angles = np.array(angles, dtype=np.int64)
    angles_ok = np.array(angles_ok, dtype=np.int64)
    positions = np.recarray(Ntot, dtype=[('x', np.int64),('y', np.int64)])
    positions.x = positions_x
    positions.y = positions_y
        
    dev_fname = '%s/dev_CAM_N%0000d.tfrecords' % (data_dir.rstrip('/'),  Ndev)
    convert_to_tfrecords(images[dev_idx], angles[dev_idx],
                         angles_ok[dev_idx], positions[dev_idx], dev_fname)

    train_fname = '%s/train_CAM_N%000d.tfrecords' % (data_dir.rstrip('/'),  Ntrain)
    convert_to_tfrecords(images[train_idx], angles[train_idx],
                         angles_ok[train_idx], positions[train_idx], train_fname)
        
    print('\nSaved data\n%s' % ( '-'*20))
    print(' Total num:',  Ntot)
    print(' Num train:',  train_idx.shape[0])
    print(' Num dev:',  dev_idx.shape[0])
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:33:25 2016

@author: hjalmar
"""


from ht_helper import get_input, whiten
# Tensorflow has to be imported before Gtk (VideoReader)
# see https://github.com/tensorflow/tensorflow/issues/1373
#import tensorflow as tf
from video_tools import VideoReader
from matplotlib import pyplot as plt
from datetime import datetime
from scipy.misc import imresize, imsave
from scipy.spatial.distance import pdist
from time import sleep
from glob import glob
import numpy as np
import warnings
import tables
import sys

warnings.simplefilter("ignore")


class FrameStepper:

    def __init__(self, fname):
        """
        """
        self.vr = VideoReader(fname)
        self.dt = 1/self.vr.fps
        self.frame = self.vr.get_frame(0.0)
        self.t = self.vr.get_current_position(fmt='time')
        self.n = self.vr.get_current_position(fmt='frame_number')
        self.tot_n = self.vr.duration_nframes
        self.duration = self.vr.duration_seconds

    def read_t(self, t):
        """
        Read a frame at t. t has to be >= to 0,
        and <= the total video duration.
        """
        self.frame = self.vr.get_frame(t)
        self.t = self.vr.get_current_position(fmt='time')
        self.n = self.vr.get_current_position(fmt='frame_number')
        if self.n is None:
            self.n = self.t*self.vr.fps

    def read_frame(self, frame_num):
        """
        Read frame number frame_num. frame_num has to be >= to 0,
        and <= the total number of frames in video.
        """
        self.read_t(self, frame_num / self.vr.fps)

    def next(self):
        """
        Reads next frame.
        Cannot be last frame.
        """
        self.frame = self.vr.get_next_frame()
        self.t = self.vr.get_current_position(fmt='time')
        self.n = self.vr.get_current_position(fmt='frame_number')

        if self.n is None:
            self.n = self.t*self.vr.fps

    def previous(self):
        """
        Reads previous frame.
        Current frame cannot the first (i.e. t = 0 and n = 1).
        """
        if (self.t - self.dt < 0) or (self.n == 1):
            print('Frame NOT updated!\n'
                  'The current frame is already the first.\n'
                  'Previous frame does not exist.')
        else:
            self.frame = self.vr.get_frame(self.t - self.dt)
            self.t = self.vr.get_current_position(fmt='time')
            self.n = self.vr.get_current_position(fmt='frame_number')
            if self.n is None:
                self.n = self.t*self.vr.fps

    def jump_t(self, jump_dur):
        """
        Jumps some duration of time from current time.
        The jump duration plus the current time has to be less than the
        total video duration.
        """
        t1 = self.vr.get_current_position(fmt='time')
        self.frame = self.vr.get_frame(t1 + jump_dur)
        self.t = self.vr.get_current_position(fmt='time')
        self.n = self.vr.get_current_position(fmt='frame_number')
        if self.n is None:
            self.n = self.t*self.vr.fps

    def jump_nf(self, nframes):
        """
        Jump some number of frames from current frame.
        The number of frames to jump plus the current frame number needs to be
        less than the total number of frames in the video.
        """
        jump_dur = nframes*self.dt
        self.jump_t(jump_dur)

    def close(self):
        """
        Close the video.
        """
        self.vr.close()


class ManuallyAnnoteFrames:
    """
    """

    def __init__(self, video_fname, log_fname=None):
        """
        """
        # Supress the plt.ginput warning.
        warnings.warn("deprecated", DeprecationWarning)

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
        self.fstp = FrameStepper(self.video_fname)
        self.available_nf = self.fstp.tot_n
        self.frame = self.fstp.frame
        self.frame_num = self.fstp.n
        self.frame_time = self.fstp.t

        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_axes([0.01, 0.01, 0.97, 0.97])
        self.num_labelled = 0
        self.imwdt = self.frame.shape[1]
        self.imhgt = self.frame.shape[0]
        self.head_data = {'center_x': 0.0,
                          'center_y': 0.0,
                          'angle': 0.0,
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

    def run_batch(self, t0, seq_len=200, n_seqs=10):
        """
        """
        self.max_skip = self.fstp.duration - seq_len * (n_seqs - 1) * self.fstp.dt - t0
        self.max_skip /= (n_seqs - 1)
        self.available_nf - seq_len * n_seqs

        for i in range(n_seqs):
            print('Starting sequence %d of %d at time %1.2fs.' %
                  (i+1, n_seqs, t0))
            self.run_sequence(t0, seq_len=200)
            #self.fig.clf()  # Clear figure to not accumulate junk.
            if i < n_seqs - 1:
                if not get_input('Continue', bool, default=True):
                    print('Quitting without finishing batch.')
                    break
                # Start time for next sequence.
                # Random drawn from the interval [1/fps, max_skip+1/fps) s.
                t0 = self.frame_time + np.random.random() * self.max_skip + self.fstp.dt
            else:
                print('Batch complete.\n%d frames labelled.' %
                      self.num_labelled)

        self.close()

    def run_sequence(self, t0, seq_len=200):
        """
        """

        self._seq_len = seq_len
        self.fstp.read_t(t0)
        self.frame = self.fstp.frame
        self.frame_num = self.fstp.n
        self.frame_num_start = self.fstp.n
        self.frame_num_end = self.fstp.n + seq_len
        self.frame_time = self.fstp.t

        self._annote(self.frame_num-self.frame_num_start+1)
        self._write_log(write_header=False)

        for i in range(1, seq_len):
            self.fstp.next()
            self.frame = self.fstp.frame
            self.frame_num = self.fstp.n
            self.frame_time = self.fstp.t
            self._annote(self.frame_num-self.frame_num_start+1)
            if self.head_poition_ok:
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
                 'gaze_angle(rad), box_width(px), box_height(px), '
                 'forehead_pos_x(px), forehead_pos_y(px), '
                 'tuft_pos_left_x(px), tuft_pos_left_y(px), '
                 'tuft_pos_right_x(px), tuft_pos_right_y(px)\n' %
                 (self.start_t.strftime("%Y-%m-%d %H:%M:%S"),
                  self.video_fname.split('/')[-1].split('.')[-2],
                  self._head_backwrd_shift,
                  self._head_len_wdt_ratio,
                  self._head_scaling))
        else:
            s = ('%d, %1.9f, %1.2f, %1.2f, %1.9f, %1.2f, %1.2f, '
                 '%1.2f, %1.2f, %1.2f, %1.2f, %1.2f, %1.2f\n' %
                 (self.frame_num,
                  self.frame_time,
                  self.head_data['center_x'],
                  self.head_data['center_y'],
                  self.head_data['angle'],
                  self.head_data['box_width'],
                  self.head_data['box_height'],
                  self.head_data['forehead_pos_x'],
                  self.head_data['forehead_pos_y'],
                  self.head_data['tuft_pos_left_x'],
                  self.head_data['tuft_pos_left_y'],
                  self.head_data['tuft_pos_right_x'],
                  self.head_data['tuft_pos_right_y']))

        self.f.write(s)

    def _annote(self, frame_num_in_seq):
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

            s = ('start: %d, end: %d, current: %d, remaining: %d' %
                 (self.frame_num_start, self.frame_num_end,
                  frame_num_in_seq, self._seq_len - frame_num_in_seq))
            txt0 = self.ax.text(self.imwdt-5, 5, s, va='bottom',
                                ha='right', fontsize=12, color=[0.4, 0.8, 0.2])

            s = 'LEFT to SELECT 1) forehead, 2) one tuft, 3) other tuft\n'
            s += 'MIDDLE to SKIP'
            txt1 = self.ax.text(self.imwdt//2, self.imhgt-10, s, va='top',
                                color='w', fontsize=14, ha='center')
            plt.draw()
            self._get_head_pos_and_rotation()
            if self.head_poition_ok:
                draw_head_position_and_angle(self.ax,
                                             self.imwdt,
                                             self.imhgt,
                                             self._head_backwrd_shift,
                                             self.head_data)
            txt1.set_visible(False)
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
            self.head_poition_ok = True
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
            self.head_poition_ok = False

    def close(self):
        """
        """
        self.fstp.vr.close()
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

    # Get position of box around head.
    bx_wdt, bx_lngt = data['box_width'], data['box_height']
    angle = data['angle']
    # left side
    l_x = hp_x + (bx_wdt/2) * np.cos(angle + np.pi/2)
    l_y = hp_y + (bx_wdt/2) * np.sin(angle + np.pi/2)
    # back left corner
    blc_x = l_x + bx_lngt * head_backwrd_shift * np.cos(angle + np.pi)
    blc_y = l_y + bx_lngt * head_backwrd_shift * np.sin(angle + np.pi)
    # front left corner
    flc_x = l_x + bx_lngt * (1 - head_backwrd_shift) * np.cos(angle)
    flc_y = l_y + bx_lngt * (1 - head_backwrd_shift) * np.sin(angle)
    # right side
    r_x = hp_x + (bx_wdt/2) * np.cos(angle - np.pi/2)
    r_y = hp_y + (bx_wdt/2) * np.sin(angle - np.pi/2)
    # back right corner
    brc_x = r_x + bx_lngt * head_backwrd_shift * np.cos(angle - np.pi)
    brc_y = r_y + bx_lngt * head_backwrd_shift * np.sin(angle - np.pi)
    # front right corner
    frc_x = r_x + bx_lngt * (1 - head_backwrd_shift) * np.cos(angle)
    frc_y = r_y + bx_lngt * (1 - head_backwrd_shift) * np.sin(angle)

    # Plot the box around the head
    ax.plot([blc_x, flc_x], [blc_y, flc_y], '-w')
    ax.plot([brc_x, frc_x], [brc_y, frc_y], '-w')
    ax.plot([blc_x, brc_x], [blc_y, brc_y], '-w')
    ax.plot([flc_x, frc_x], [flc_y, frc_y], '-w')
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
    fstp = FrameStepper(video_fname)
    frame = fstp.frame

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.01, 0.01, 0.97, 0.97])
    imwdt = frame.shape[1]
    imhgt = frame.shape[0]

    log_data, log_header = read_log_data(log_fname)

    for i, dat in enumerate(log_data):

        fstp.read_t(dat['frame_time'])
        ax.cla()
        ax.imshow(fstp.frame, origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        draw_head_position_and_angle(ax, imwdt, imhgt, head_backwrd_shift, dat)
        plt.draw()
        fig.savefig('test_%d.png' % i)
        sleep(5)

    fstp.close()


def train_data_to_storage(log_fname,
                          video_dir,
                          out_fname,
                          mode='a',
                          head_size=200,
                          patch_size=32,
                          max_patch_overlap=0.25,
                          num_neg_per_pos=2,
                          debug=False):
    """
    Parameters
    ----------
    log_fname
    video_dir
    out_fname
    head_size
    patch_size
    max_patch_overlap
    num_neg_per_pos     : The number of negative examples per positive example.
                          The locations of negative examples are drawn from the 
                          locations of postive examples (head positions), 
                          but extracted from non-matching frames.
    debug
    """
    
    if debug:
        if num_neg_per_pos > 4:
            print('Max for negative images can be displayed in debug mode.\n'
                   'Set "max_neg_per_pos" <= 4.')
            return 0
        import pdb
        plt.ion()
        #fig = plt.figure(figsize=[23.1875, 6.125])
        axs = []
        if num_neg_per_pos > 2:
            fig = plt.figure(figsize=[11.6, 12.425])
            
        else:
            fig = plt.figure(figsize=[11.6, 8.425])
            axs.append(fig.add_axes([0.005, 0.50, 0.56, 0.49]))
            axs.append(fig.add_axes([0.501, 0.50, 0.49, 0.49]))   
            axs.append(fig.add_axes([0.005, 0.001, 0.49, 0.49]))
            axs.append(fig.add_axes([0.501, 0.001, 0.49, 0.49]))
        if num_neg_per_pos > 2:
            axs.append(fig.add_axes([0.005, 0.66, 0.54, 0.32]))
            axs.append(fig.add_axes([0.501, 0.66, 0.49, 0.32]))   
            axs.append(fig.add_axes([0.005, 0.33, 0.49, 0.32]))
            axs.append(fig.add_axes([0.501, 0.33, 0.49, 0.32]))            
            axs.append(fig.add_axes([0.005, 0.01, 0.49, 0.32]))
            axs.append(fig.add_axes([0.501, 0.01, 0.49, 0.32]))       
             
    scale_factor = patch_size / head_size
    min_dist_to_pos = patch_size * max_patch_overlap

    log_data, log_header = read_log_data(log_fname)
    video_fname = '%s/%s*' % (video_dir.rstrip('/'), log_header['video_fname'])
    video_fname = glob(video_fname)
    if len(video_fname) != 1:
        print('The video file matching to the log cannot be found in %s'
              % video_dir)
        print('Found:', video_fname)
        return 0
    fstp = FrameStepper(video_fname[0])
    frame = whiten(imresize(fstp.frame, scale_factor))

    # HDF5 storage of the data
    f = tables.openFile(out_fname, mode=mode)
    if mode == 'w':  # Write new file -- OVERWRITES!
        atom_data = tables.Atom.from_dtype(frame.dtype)
        atom_labels = tables.Atom.from_dtype(np.dtype(int))
        filters = tables.Filters(complib='blosc', complevel=5)

        # TODO: remove pos_data, neg_data and head_angles
        pos_data = f.createEArray(f.root, 'heads', atom_data,
                                  shape=(patch_size, patch_size, 0),
                                  filters=filters,
                                  chunkshape=(patch_size, patch_size, 1))
        head_angles = f.createEArray(f.root, 'head_angles', atom_labels,
                                     shape=(1, 0),
                                     filters=filters,
                                     chunkshape=(1, 1))
        neg_data = f.createEArray(f.root, 'no_heads', atom_data,
                                  shape=(patch_size, patch_size, 0),
                                  filters=filters,
                                  chunkshape=(patch_size, patch_size, 1))

        data = f.createEArray(f.root, 'data', atom_data,
                              shape=(0, patch_size, patch_size),
                              filters=filters,
                              chunkshape=(patch_size, patch_size, 1))
        labels = f.createEArray(f.root, 'labels', atom_labels,
                                shape=(0, 2),
                                filters=filters,
                                chunkshape=(1, 2))                              
    elif mode == 'a':
        ## Open existing file to append data.
        pos_data = f.root.heads
        head_angles = f.root.head_angles
        neg_data = f.root.no_heads
        data = f.root.data
        labels = f.root.labels

    # Scale head positions and add padding
    #import pdb
    #pdb.set_trace()
    log_data['center_x'] = log_data['center_x'] * scale_factor
    log_data['center_y'] = log_data['center_y'] * scale_factor
    # Head angles to degrees (-180 to 180)
    log_data['angle'] = (360 * (log_data['angle'] / np.pi)).round()

    def extract_patch(x, y, frm):
        """
        """
        row0 = int(y - patch_size//2)
        row1 = int(row0 + patch_size)
        col0 = int(x - patch_size//2)
        col1 = int(col0 + patch_size)
        r0, r1 = max(row0, 0), min(row1, frm.shape[0])
        c0, c1 = max(col0, 0), min(col1, frm.shape[1])
        # White (std=1, mean=0) padding, to keep w patch whitening.
        patch = np.random.standard_normal(size=(patch_size, patch_size))
        pr0, pr1 = int(r0 - row0), int(patch_size - (row1 - r1))
        pc0, pc1 = int(c0 - col0), int(patch_size - (col1 - c1))
        patch[pr0: pr1, :][:, pc0: pc1] = whiten(frm[r0:r1][:, c0:c1])
        if debug:
            clr = 'r'
            axs[0].plot([c0, c1], [r0, r0], clr)
            axs[0].plot([c0, c1], [r1, r1], clr)
            axs[0].plot([c0, c0], [r0, r1], clr)
            axs[0].plot([c1, c1], [r0, r1], clr)
        return patch

    for i, dat in enumerate(log_data):
        # Just a counter printed on the command line
        s = '%04d' % (log_data.shape[0] - i)
        sys.stdout.write(s)
        sys.stdout.flush()
        sys.stdout.write('\b'*len(s))
        # Read the frame.
        frame_num = dat['frame_num']
        fstp.read_t(dat['frame_time'])
        frame = imresize(fstp.frame.mean(axis=2), scale_factor)
        
        if debug:
            axs[0].cla()
            axs[0].imshow(frame, origin='lower', cmap=plt.cm.gray)
            axs[0].set_xlim([0, frame.shape[1]])
            axs[0].set_ylim([0, frame.shape[0]])
             
        # Positive example, i.e. head
        pos_x, pos_y = dat['center_x'], dat['center_y']
        pos_patch = extract_patch(pos_x, pos_y, frame)
        
        # TODO: remove pos_data
        pos_data.append(pos_patch.reshape((patch_size, patch_size, 1)))

        data.append(pos_patch.reshape((1, patch_size, patch_size)))
        # head angle
        # TODO: remove head angle
        head_angles.append(np.array(dat['angle'], dtype=int).reshape((1, 1)))
        
        labels.append(np.array([1, dat['angle']], dtype=int).reshape((1, 2)))

        # Negative examples, i.e. background/no head
        idx = np.random.randint(0, log_data.shape[0], num_neg_per_pos)
        all_neg_x, all_neg_y = log_data['center_x'][idx], log_data['center_y'][idx]
        ndists = num_neg_per_pos + num_neg_per_pos * (num_neg_per_pos+ - 1) - np.arange(num_neg_per_pos).sum()
        #print(ndists)
        ok = np.zeros(ndists, dtype=bool)        
        intradists_x = pdist(all_neg_x.reshape(num_neg_per_pos, 1))
        intradists_y = pdist(all_neg_y.reshape(num_neg_per_pos, 1))
        ok[:num_neg_per_pos] = np.logical_and(np.abs(all_neg_x - pos_x) > min_dist_to_pos,
                                              np.abs(all_neg_y - pos_y) > min_dist_to_pos)
        ok[num_neg_per_pos:] = np.logical_and(intradists_x > min_dist_to_pos,
                                              intradists_y > min_dist_to_pos)
        while not ok.all():   # Check that the randomly drawn position are not
                              # within head position in current frame.
            idx = np.random.randint(0, log_data.shape[0], num_neg_per_pos)
            all_neg_x, all_neg_y = log_data['center_x'][idx], log_data['center_y'][idx]
            intradists_x = pdist(all_neg_x.reshape(num_neg_per_pos, 1))
            intradists_y = pdist(all_neg_y.reshape(num_neg_per_pos, 1))
            ok[:num_neg_per_pos] = np.logical_and(np.abs(all_neg_x - pos_x) > min_dist_to_pos,
                                                  np.abs(all_neg_y - pos_y) > min_dist_to_pos)
            ok[num_neg_per_pos:] = np.logical_and(intradists_x > min_dist_to_pos,
                                                  intradists_y > min_dist_to_pos)

        j = 2
        for neg_x, neg_y in zip(all_neg_x, all_neg_y):
            neg_patch = extract_patch(neg_x, neg_y, frame)
            # TODO: remove neg_data
            neg_data.append(neg_patch.reshape((patch_size, patch_size, 1)))
            labels.append(np.array([0, 0], dtype=int).reshape((1, 2)))
            data.append(neg_patch.reshape((1, patch_size, patch_size)))
            if debug and j < 6:
                axs[j].imshow(neg_patch, origin='lower', cmap=plt.cm.gray)
                j += 1

        if debug:
            axs[0].plot(all_neg_x, all_neg_y, 'or')
            axs[0].plot(pos_x, pos_y, 'og')
            axs[1].imshow(pos_patch, origin='lower', cmap=plt.cm.gray)
            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])
            plt.draw()
            pdb.set_trace()


    if debug:
        plt.ioff()

    fstp.close()
    f.close()


def check_training_data(fname, n):
    """
    Visually check the training data.
    """
    f = tables.openFile(fname.encode('utf-8'), 'r')

    plt.ion()
    fig = plt.figure(figsize=[23.1875,   9.55])
    ax0 = fig.add_axes([0.005, 0.01, 0.49, 0.97])
    ax1 = fig.add_axes([0.501, 0.01, 0.49, 0.97])

    # TODO: adapt to new data format.
    N = min(f.root.heads.shape[2], f.root.no_heads.shape[2])
    if (N <= n):
        print('Max value of parameter "n" is %d' % N)
        return 0
    else:
        max_i = min(N, n + 300)
    heads = f.root.heads[:, :, n: max_i]
    angles = f.root.head_angles[0,  n: max_i]
    no_heads = f.root.no_heads[:, :, n: max_i]

    def tile_images(A3d, shape):
        h, w, n = A3d.shape
        im = np.empty(shape, dtype=A3d.dtype)
        nr = shape[0]//h
        nc = shape[1]//w
        i = 0
        XY = []
        for r in range(nr):
            if i < n:
                for c in range(nc):
                    if i < n:
                        im[r * h: r * h + h, c * w: c * w + h] = A3d[:, :, i]
                        XY.append((c * w, r * h))
                        i += 1
                    else:
                        break
            else:
                break
        return im, XY

    heads, XY = tile_images(heads, (480, 640))
    no_heads, _ = tile_images(no_heads, (480, 640))

    f.close()

    ax0.imshow(heads, origin='lower', cmap=plt.cm.gray)
    for i, xy in enumerate(XY):
        ax0.text(xy[0], xy[1], i+n, va='bottom', ha='left', fontsize=7)
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1.imshow(no_heads, origin='lower', cmap=plt.cm.gray)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    plt.draw()
    plt.ioff()
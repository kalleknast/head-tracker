# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 18:46:36 2016

@author: hjalmar
"""
#import termios
#import fcntl
#import sys
#import os
import numpy as np
# Tensorflow has to be imported before Gtk (VideoReader)
# see https://github.com/tensorflow/tensorflow/issues/1373
#import tensorflow as tf
from video_tools import VideoReader
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import ipdb


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
    
    
def softmax(logits):
    """
    Temporary replacement for tf.nn.softmax()
    See issue #2810 on github.com/tensorflow/tensorflow/issues  
    """
    e = np.exp(logits)
    return e / e.sum(axis=1,  keepdims=True)    
    

def get_window(image,  win_sz,  center):
    """
    center - (r, c)
    """    
    im_nr, im_nc = image.shape  # number of rows, columns in image
    w_sz = win_sz
    #centers = self.levels[k].centers
    win = np.empty((win_sz,  win_sz))
    ir0 = int(round(center[0]-w_sz/2))
    ir1 = int(round(center[0]+w_sz/2))
    ic0 = int(round(center[1]-w_sz/2))
    ic1 = int(round(center[1]+w_sz/2))

    if (ir0 < 0) and (ic0 < 0):
        tmp = image[:ir1, :ic1]
        # Whiten
        adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
        # subtract mean and divide by standard deviation
        win[-ir0:, -ic0:] = (tmp - tmp.mean()) / adj_std
        # pad with white noise
        win[:, :-ic0] = np.random.standard_normal(size=(w_sz, -ic0))
        win[:-ir0, :] = np.random.standard_normal(size=(-ir0, w_sz))
    elif (ir0 < 0) and ((ic0 >= 0) and (ic1 <= im_nc)):
        tmp = image[:ir1, ic0:ic1]
        adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
        win[-ir0:, :] = (tmp - tmp.mean()) / adj_std
        win[:-ir0, :] = np.random.standard_normal(size=(-ir0, w_sz))
    elif (ir0 < 0) and (ic1 > im_nc):
        tmp = image[:ir1, ic0:]
        adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
        win[-ir0:, :im_nc-ic0] = (tmp - tmp.mean()) / adj_std
        win[:, im_nc-ic0:] = np.random.standard_normal(size=(w_sz, w_sz-tmp.shape[1]))
        win[:-ir0, :] = np.random.standard_normal(size=(-ir0, w_sz))
    elif ((ir0 >= 0) and (ir1 <= im_nr)) and (ic0 < 0):
        tmp = image[ir0:ir1, :ic1]
        adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
        win[:, -ic0:] = (tmp - tmp.mean()) / adj_std
        win[:, :-ic0] = np.random.standard_normal(size=(w_sz, -ic0))
    elif ((ir0 >= 0) and (ir1 <= im_nr)) and ((ic0 >= 0) and (ic1 <= im_nc)):
        tmp = image[ir0:ir1, ic0:ic1]
        adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
        win[:, :] = (tmp - tmp.mean()) / adj_std
    elif ((ir0 >= 0) and (ir1 <= im_nr)) and (ic1 > im_nc):
        tmp = image[ir0:ir1, ic0:]
        adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
        win[:, :im_nc-ic0] = (tmp - tmp.mean()) / adj_std
        win[:, im_nc-ic0:] = np.random.standard_normal(size=(w_sz, w_sz-tmp.shape[1]))
    elif (ir1 > im_nr) and (ic0 < 0):
        tmp = image[ir0:, :ic1]
        adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
        win[:im_nr-ir0, -ic0:] = (tmp - tmp.mean()) / adj_std
        win[:, :-ic0] = np.random.standard_normal(size=(w_sz, -ic0))
        win[im_nr-ir0:, :] = np.random.standard_normal(size=(w_sz-tmp.shape[0], w_sz))
    elif (ir1 > im_nr) and((ic0 >= 0) and (ic1 <= im_nc)):
        tmp = image[ir0:, ic0:ic1]
        adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
        win[:im_nr-ir0, :] = (tmp - tmp.mean()) / adj_std
        win[im_nr-ir0:, :] = np.random.standard_normal(size=(w_sz-tmp.shape[0], w_sz))
    elif (ir1 > im_nr) and (ic1 > im_nc):
        tmp = image[ir0:, ic0:]
        adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
        win[:im_nr-ir0, :im_nc-ic0] = (tmp - tmp.mean()) / adj_std
        win[:, im_nc-ic0:] = np.random.standard_normal(size=(w_sz, w_sz-tmp.shape[1]))
        win[im_nr-ir0:, :] = np.random.standard_normal(size=(w_sz-tmp.shape[0], w_sz))
    
    return win

#
#def whiten(X):
#    """
#    As tensorflow.image.per_image_whitening(image)
#    https://www.tensorflow.org/versions/r0.7/api_docs/python/image.html#per_image_whitening
#    """
#    if X.ndim == 3:
#        Y = X.reshape((X.shape[0], -1))
#        m = Y.mean(axis=1)
#        adjusted_std = np.concatenate((Y.std(axis=1),
#                                       np.repeat(1.0/np.sqrt(Y.shape[1]),
#                                                 Y.shape[0])))
#        adjusted_std = adjusted_std.reshape((-1, 2)).max(axis=1)   
#        adjusted_std = adjusted_std.repeat(np.prod(X.shape[-2:])).reshape(X.shape)
#        m = m.repeat(np.prod(X.shape[-2:])).reshape(X.shape)        
#    else:
#        adjusted_std = max(X.std(), 1.0/np.sqrt(X.size))
#        m = X.mean()
#
#    return (X - m) / adjusted_std
#

def get_input(q_str, ans_type, default=None, check=False):
    """
    Parameters
    ----------
    q_str       -- Question/input string, e.g.:
                   ' Are the stimulus positions OK'
                   ' 1st scan --- Enter diameter of the UV stimulation spot (micron)'
    ans_type    -- Type of the returned answer, e.g. int, float, str, bool
    default     -- Default value for output, if given it should have the same
                   type as ans_type. For not defalut value enter None.
    check       -- Whether or not to ask if the input value is ok?

    Return
    ------
    ans         -- Reply to asked question, same type as ans_type

    Example
    -------
    ans = get_input( ' Enter duration of UV stimulation (ms)', int, 50, True )
    """

    q_str = ' ' + q_str

    if type(default) in [float, int, str]:
        q_str += ': [' + str(default) + ']\n'
    elif type(default) is bool:
        if default:
            q_str += ' ([y]/n)?\n'
        else:
            q_str += ' (y/[n])?\n'
    elif default is None:
        q_str += ':\n'

    OK = False
    while not OK:

        s = input(q_str)
        if not s:
            ans = default
            OK = True
        else:
            if ans_type is bool:
                if s.lower() == 'n':
                    ans, OK = False, True
                elif s.lower() == 'y':
                    ans, OK = True, True
                else:
                    OK = False
            elif ans_type is str:
                if s.isalpha():
                    ans, OK = s, True
                else:
                    print(' Invalid input')
                    OK = False
            else:
                try:
                    ans = ans_type(s)
                    OK = True
                except:
                    print(' Invalid input')
                    OK = False

        if check:
            ok = 'x'
            while not ok.lower() in ['y', 'n', '']:
                ok = input(str(ans) + ' OK ([y]/n)?\n')
            if ok.lower() == 'n':
                OK = False

    return ans


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


#def window_level0(im):
#    """
#    """
#
#    if im.shape != (76, 102):
#        print('im needs to have shape == (76, 108)')
#        return 0
#
#    # possibly 48, should give 
#    w_sz = 40
#    step = int(w_sz/2)
#    im_nr, im_nc = im.shape
#    offset_c = -int(w_sz*(np.ceil(im_nc/w_sz)-(im_nc/w_sz))/2)
#    offset_r = -int(w_sz*(np.ceil(im_nr/w_sz)-(im_nr/w_sz))/2)
#    
#    im_r0 = [offset_r, offset_r, offset_r, offset_r, offset_r,                   # 1st row
#             offset_r+step, offset_r+step, offset_r+step,       # 2nd row
#             offset_r+2*step, offset_r+2*step, offset_r+2*step, offset_r+2*step, offset_r+2*step] # 3rd and last row
#             
#    im_c0 = [offset_c, offset_c+step, offset_c+2*step, offset_c+3*step, offset_c+4*step,  # cols in 1st row
#             offset_c, offset_c+2*step, offset_c+4*step,                  # cols in 2nd row
#             offset_c, offset_c+step, offset_c+2*step, offset_c+3*step, offset_c+4*step]  # cols in 3rd row 
#    
#    wins = np.empty((len(im_r0), w_sz, w_sz))
#    centers = np.empty((len(im_r0), 2))
#
#    i = 0
#    for ir0, ic0 in zip(im_r0, im_c0):
#
#        ir1 = ir0 + w_sz            
#        ic1 = ic0 + w_sz
#        centers[i,:] = ir0 + w_sz//2, ic0 + w_sz//2
#        win = wins[i, :, :]
#
#        if (ir0 < 0) and (ic0 < 0):
#            tmp = im[:ir1, :ic1]
#            # Whiten
#            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
#            # subtract mean and divide by standard deviation
#            wins[i, -offset_r:, -offset_c:] = (tmp - tmp.mean()) / adj_std
#            # pad with white noise
#            wins[i, :, :-offset_c] = np.random.standard_normal(size=(w_sz, -offset_c))
#            wins[i, :-offset_r, :] = np.random.standard_normal(size=(-offset_r, w_sz))
#        elif (ir0 < 0) and ((ic0 >= 0) and (ic1 <= im_nc)):
#            tmp = im[:ir1, ic0:ic1]
#            # Whiten
#            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
#            # subtract mean and divide by standard deviation
#            wins[i, -offset_r:, :] = (tmp - tmp.mean()) / adj_std
#            # pad with white noise
#            wins[i, :-offset_r, :] = np.random.standard_normal(size=(-offset_r, w_sz))
#        elif (ir0 < 0) and (ic1 > im_nc):
#            tmp = im[:ir1, ic0:]
#            # Whiten
#            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
#            # subtract mean and divide by standard deviation
#            wins[i, -offset_r:, :offset_c] = (tmp - tmp.mean()) / adj_std
#            # pad with white noise
#            wins[i, :, offset_c:] = np.random.standard_normal(size=(w_sz, -offset_c))
#            wins[i, :-offset_r, :] = np.random.standard_normal(size=(-offset_r, w_sz))
#        elif ((ir0 >= 0) and (ir1 <= im_nr)) and (ic0 < 0):
#            tmp = im[ir0:ir1, :ic1]
#            # Whiten
#            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
#            # subtract mean and divide by standard deviation
#            wins[i, :, -offset_c:] = (tmp - tmp.mean()) / adj_std
#            # pad with white noise
#            wins[i, :, :-offset_c] = np.random.standard_normal(size=(w_sz, -offset_c))
#        elif ((ir0 >= 0) and (ir1 <= im_nr)) and ((ic0 >= 0) and (ic1 <= im_nc)):
#            tmp = im[ir0:ir1, ic0:ic1]
#            # Whiten
#            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
#            # subtract mean and divide by standard deviation
#            wins[i, :, :] = (tmp - tmp.mean()) / adj_std
#        elif ((ir0 >= 0) and (ir1 <= im_nr)) and (ic1 > im_nc):
#            tmp = im[ir0:ir1, ic0:]
#            # Whiten
#            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
#            # subtract mean and divide by standard deviation
#            wins[i, :, :offset_c] = (tmp - tmp.mean()) / adj_std
#            # pad with white noise
#            wins[i, :, offset_c:] = np.random.standard_normal(size=(w_sz, -offset_c))
#        elif (ir1 > im_nr) and (ic0 < 0):
#            tmp = im[ir0:, :ic1]
#            # Whiten
#            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
#            # subtract mean and divide by standard deviation
#            wins[i, :offset_r, -offset_c:] = (tmp - tmp.mean()) / adj_std
#            # pad with white noise
#            wins[i, :, :-offset_c] = np.random.standard_normal(size=(w_sz, -offset_c))
#            wins[i, offset_r:, :] = np.random.standard_normal(size=(-offset_r, w_sz))
#        elif (ir1 > im_nr) and((ic0 >= 0) and (ic1 <= im_nc)):
#            tmp = im[ir0:, ic0:ic1]
#            # Whiten
#            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
#            # subtract mean and divide by standard deviation
#            win[:offset_r, :] = (tmp - tmp.mean()) / adj_std
#            # pad with white noise
#            wins[i, offset_r:, :] = np.random.standard_normal(size=(-offset_r, w_sz))
#        elif (ir1 > im_nr) and (ic1 > im_nc):
#            tmp = im[ir0:, ic0:]
#            # Whiten
#            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
#            # subtract mean and divide by standard deviation
#            wins[i, :offset_r, :offset_c] = (tmp - tmp.mean()) / adj_std
#            # pad with white noise
#            wins[i, :, offset_c:] = np.random.standard_normal(size=(w_sz, -offset_c))
#            wins[i, offset_r:, :] = np.random.standard_normal(size=(-offset_r, w_sz))
#        i += 1
#
#    return wins, centers
#
#
#def window_level1(im, center):
#    """
#    """
#    
#    if im.shape != (76, 102):
#        print('im needs to have shape == (76, 108)')
#        return 0
#
#    level0_step = 20
#    w_sz = 32
#    subim_sz = level0_step + w_sz
#    step = int(level0_step/2)
#    im_nr, im_nc = im.shape
#
#    offset_r = int(round(center[0] - subim_sz/2))
#    offset_c = int(round(center[1] - subim_sz/2))
#
#    im_r0 = [offset_r, offset_r, offset_r,
#             step+offset_r, step+offset_r, step+offset_r,
#             2*step+offset_r, 2*step+offset_r, 2*step+offset_r]
#    im_c0 = [offset_c, step+offset_c, 2*step+offset_c,
#             offset_c, step+offset_c, 2*step+offset_c,
#             offset_c, step+offset_c, 2*step+offset_c]
#    
#    wins = np.empty((len(im_r0), w_sz, w_sz))
#    centers = np.empty((len(im_r0), 2))
#
#    i = 0
#    for ir0, ic0 in zip(im_r0, im_c0):
#
#        ir1 = ir0 + w_sz            
#        ic1 = ic0 + w_sz
#        centers[i,:] = ir0 + w_sz//2, ic0 + w_sz//2
#        win = wins[i, :, :]
#
#        if (ir0 < 0) and (ic0 < 0):
#            tmp = im[:ir1, :ic1]
#            # Whiten
#            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
#            # subtract mean and divide by standard deviation
#            wins[i, -ir0:, -ic0:] = (tmp - tmp.mean()) / adj_std
#            # pad with white noise
#            wins[i, :, :-ic0] = np.random.standard_normal(size=(w_sz, -ic0))
#            wins[i, :-ir0, :] = np.random.standard_normal(size=(-ir0, w_sz))
#        elif (ir0 < 0) and ((ic0 >= 0) and (ic1 <= im_nc)):
#            tmp = im[:ir1, ic0:ic1]
#            # Whiten
#            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
#            # subtract mean and divide by standard deviation
#            wins[i, -ir0:, :] = (tmp - tmp.mean()) / adj_std
#            # pad with white noise
#            wins[i, :-ir0, :] = np.random.standard_normal(size=(-ir0, w_sz))
#        elif (ir0 < 0) and (ic1 > im_nc):
#            tmp = im[:ir1, ic0:]
#            # Whiten
#            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
#            # subtract mean and divide by standard deviation
#            wins[i, -ir0:, :im_nc-ic0] = (tmp - tmp.mean()) / adj_std
#            # pad with white noise
#            wins[i, :, im_nc-ic0:] = np.random.standard_normal(size=(w_sz, w_sz-tmp.shape[1]))
#            wins[i, :-ir0, :] = np.random.standard_normal(size=(-ir0, w_sz))
#        elif ((ir0 >= 0) and (ir1 <= im_nr)) and (ic0 < 0):
#            tmp = im[ir0:ir1, :ic1]
#            # Whiten
#            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
#            # subtract mean and divide by standard deviation
#            wins[i, :, -ic0:] = (tmp - tmp.mean()) / adj_std
#            # pad with white noise
#            wins[i, :, :-ic0] = np.random.standard_normal(size=(w_sz, -ic0))
#        elif ((ir0 >= 0) and (ir1 <= im_nr)) and ((ic0 >= 0) and (ic1 <= im_nc)):
#            tmp = im[ir0:ir1, ic0:ic1]
#            # Whiten
#            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
#            # subtract mean and divide by standard deviation
#            wins[i, :, :] = (tmp - tmp.mean()) / adj_std
#        elif ((ir0 >= 0) and (ir1 <= im_nr)) and (ic1 > im_nc):
#            tmp = im[ir0:ir1, ic0:]
#            # Whiten
#            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
#            # subtract mean and divide by standard deviation
#            wins[i, :, :im_nc-ic0] = (tmp - tmp.mean()) / adj_std
#            # pad with white noise
#            wins[i, :, im_nc-ic0:] = np.random.standard_normal(size=(w_sz, w_sz-tmp.shape[1]))
#        elif (ir1 > im_nr) and (ic0 < 0):
#            tmp = im[ir0:, :ic1]
#            # Whiten
#            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
#            # subtract mean and divide by standard deviation
#            wins[i, :im_nr-ir0, -ic0:] = (tmp - tmp.mean()) / adj_std
#            # pad with white noise
#            wins[i, :, :-ic0] = np.random.standard_normal(size=(w_sz, -ic0))
#            wins[i, im_nr-ir0:, :] = np.random.standard_normal(size=(w_sz-tmp.shape[0], w_sz))
#        elif (ir1 > im_nr) and((ic0 >= 0) and (ic1 <= im_nc)):
#            tmp = im[ir0:, ic0:ic1]
#            # Whiten
#            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
#            # subtract mean and divide by standard deviation
#            win[:im_nr-ir0, :] = (tmp - tmp.mean()) / adj_std
#            # pad with white noise
#            wins[i, im_nr-ir0:, :] = np.random.standard_normal(size=(w_sz-tmp.shape[0], w_sz))
#        elif (ir1 > im_nr) and (ic1 > im_nc):
#            tmp = im[ir0:, ic0:]
#            # Whiten
#            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
#            # subtract mean and divide by standard deviation
#            wins[i, :im_nr-ir0, :im_nc-ic0] = (tmp - tmp.mean()) / adj_std
#            # pad with white noise
#            wins[i, :, im_nc-ic0:] = np.random.standard_normal(size=(w_sz, w_sz-tmp.shape[1]))
#            wins[i, im_nr-ir0:, :] = np.random.standard_normal(size=(w_sz-tmp.shape[0], w_sz))
#        i += 1
#    
#    return wins, centers

def closest_coordinate(r, c, coordinates):
    """
    Parameters
    ----------------
    r                    - row index
    c                   - column index
    coordinates  - 2-d numpy.array of row and column indexes.
                   centers[:, 0] -- row indexes
                   centers[:, 1] -- columns indexes
                   Eg np.array([[r0, c0], [r1, c1], [r2, c2]])
    Returns
    -----------
    c        -    the center/coordinates closest to (r, c)
    i        -    index of c in centers
    """
    
    i = np.argmin(np.abs(coordinates - [r, c]).sum(axis=1))
    c = coordinates[i, :]
    
    return c, i


def get_max_gaze_line(angle, x, y, im_w, im_h, margin=10, units='deg'):
    """
    Returns the end positions of a line starting at x, y and with an angle of
    angle within a rectangular image with width im_w and height im_h.
    
    Use to draw the line of gaze in an image.
    
    Bottom of image is y = 0, ie not y is not equal row in an array.
    Use imshow origin='lower'
    
    Parameters
    ----------
    angle   : angle of line
    x       : x-coordinate of the line's origin
    y       : y-coordinate of the line's origin
    im_w    : width of limiting rectangle/image
    im_h    : height of limiting rectangle/image
    units   : 'deg' or 'rad'
    margin  : gazeline stops at margin

    Returns
    -------
    x1      : x-coordinate of the line's end point
    y1      : y coordinate of the line's end point
    """
        
    if units == 'deg':
        angle = np.deg2rad(angle)

    # make sure the angle stays between -pi and pi
    angle = np.arctan2(np.sin(angle), np.cos(angle))        
    
    if np.abs(angle) > np.pi/2:
        dx = x - margin
    else:
        dx = im_w - margin - x
        
    if angle > 0.0:
        dy = im_h - margin - y
    else:
        dy = y - margin
    
    # Chose the shortest radius since the longest will go outside of 
    # im
    r = min(np.abs(dx/np.cos(angle)), np.abs(dy/np.sin(angle)))

    x1 = r * np.cos(angle) + x
    y1 = r * np.sin(angle) + y
    
    return x1, y1

    
def get_gaze_line(angle, x, y, r, units='rad'):
    """
    Parameters
    ----------
    angle
    x
    y
    r
    units : Units of "angle", "rad" for radians or "deg" for degrees.
    
    Returns
    -------
    gzl_x
    gzl_y
    """  
    if units == 'rad':
        gzl_x = [x, x + r * np.cos(angle)]
        gzl_y = [y, y + r * np.sin(angle)]
    elif units == 'deg':
        gzl_x = [x, x + r * np.cos(np.deg2rad(angle))]
        gzl_y = [y, y + r * np.sin(np.deg2rad(angle))]
    else:
        raise ValueError('"units" has to be "rad" or "deg"')
    
    return gzl_x, gzl_y    
    
    
def anglediff(angles0, angles1, units='deg'):
    
    if units == 'deg':
        d = np.deg2rad(angles0) - np.deg2rad(angles1)
        adiff = np.rad2deg(np.arctan2(np.sin(d), np.cos(d)))
    elif units == 'rad':
        d = angles0 - angles1
        adiff = np.arctan2(np.sin(d), np.cos(d))
    else:
        raise ValueError('"units" has to be "rad" or "deg"')        
        
    return adiff
        
    
def angles2complex(angles):   
    """ 
    Angles in degrees, array or scalar
    """ 
    z = np.cos(np.deg2rad(angles)) + np.sin(np.deg2rad(angles))*1j
    
    return z
    

def complex2angles(z):   
    """ 
    z complex array or scalar 
    angles in degrees, array or scalar
    """ 
    angles = np.rad2deg(np.arctan2(z.imag, z.real))
    return angles
    
    
def angle2class(angles, Nclass, angles_ok=None, units='deg'):
    """
    Arguments
    ---------
    angles      - A scalar or a vector of angles
    Nclass      - Number of classes to divide angles into.
    angles_ok   - A bool, scalar or vector. True if the corresponding angle
                  is ok, ie, the head orientation in the horizontal plane was
                  clearly visible, False otherwise.
                  If given, then class Nclass-1 will code angle not ok.
    units       - Unit of angles, "deg" or "rad"
    
    Returns
    -------
    y           - A scalar or vector of same length as angles, with values that
                  run from 0 to Nclass-1. If angles_ok was given then class
                  Nclass-1 will code angle not ok.
    """
    
    if units == 'deg':
        a = np.deg2rad(angles)
    elif units == 'rad':
        a = angles
    else:
        raise ValueError('"units" has to be "rad" or "deg"')             

    # angles range -pi:pi -> shift to pi:pi
    angles = (np.arctan2(np.sin(a), np.cos(a)) + np.pi) / (2 * np.pi)
    
    
    if angles_ok is None:
        y = np.int32(Nclass * angles)  # Bin 2pi rad into Nclass bins with values 0:Nclass-1.
        if np.isscalar(y):
            if y == Nclass:
                y = 0
        else:
            y[y == Nclass] = 0  # 0 & 2pi are same angle
    
    else:
        y = np.int32((Nclass-1) * angles)
        if np.isscalar(y):
            if y == (Nclass-1):
                y = 0
            if ~ angle_ok:
                y = Nclass - 1
        else:
            y[y == Nclass] = 0
            # No clear head orientation in the horiz plane,
            # ie angle_ok = 0, is coded as the last class.
            y[~ angles_ok] = Nclass - 1
        
    return np.float32(y)
    
    
def class2angle(y,  Nclass):
    """
    Angles are from -180 to 180
    y run from 0 to Nclass-1
    """
    angles = y * (360 / Nclass) + 180 / Nclass # angles run from -180:180
    return angles - 180

    
def dist2coordinates(r, c, coordinates):
    """
    Returns the coordinate with minimum distance to r & c,
    and those coordinates with distances within 129% of the minimum distance.
    
    129% is chosen because the smallest patch is 32x32 and the overlap is 50%.
    If the head is located on the line between two neighboring patches the 
    longest distance to the closest patch will be 8 (equidistant between the 2).
    The ratio of the distances will be 8/8 = 1. If the head but shifted 1 element
    (ie the smallest possible shift) towards one of them, the ratio of their 
    distances will be (8+1)/(8-1) = 1.2857... (if the head is shifted 2 elements
    the ratio will be (8+2/(8-2) = 1.666... ).
    In the cases where 

    Parameters
    ----------------
    r                    - row index
    c                   - column index
    coordinates  - 2-d numpy.array of row and column indexes.
                   centers[:, 0] -- row indexes
                   centers[:, 1] -- columns indexes
                   Eg np.array([[r0, c0], [r1, c1], [r2, c2]])
    Returns
    d
    i          - index of d in coordinates.
    -----------

    """

    dists = np.sqrt(((coordinates - [r,  c])**2).sum(axis=1))
    sorted_idx = np.argsort(dists)
    dists = dists[sorted_idx]
    min_dist = dists.min()
    if min_dist == 0:
        norm_dists = dists/1
    else:
        norm_dists = dists/dists.min()        

    #d = dists[norm_dists < 1.67]
    #i = sorted_idx[norm_dists < 1.67]
    d = dists[norm_dists < 1.3]
    i = sorted_idx[norm_dists < 1.3]    
    
    return d,  i
    
    
def window_image(image, win_sz=48, nwin_c=4, nwin_r=3, step=24):
    """
    """

    im_nr, im_nc = image.shape
   
    offset_c = int(0.5*(im_nc - (nwin_c+1)*win_sz/2))
    offset_r = int(0.5*(im_nr - (nwin_r+1)*win_sz/2))

    im_r0 = [offset_r] * nwin_c # 1st row
    im_r0.extend([offset_r + step] * nwin_c) # 2nd row
    im_r0.extend([offset_r + 2 * step] * nwin_c) # 2nd row
    
    im_c0 = [offset_c, offset_c + step,
             offset_c + 2 * step, offset_c + 3 * step] * nwin_r
             
    nwin = int(nwin_r * nwin_c)
    centers = np.empty((nwin, 2))
    wins = np.empty((nwin, win_sz, win_sz))                 
 
    w_sz = win_sz
    
    i = 0
    for ir0, ic0 in zip(im_r0, im_c0):

        ir1 = ir0 + w_sz            
        ic1 = ic0 + w_sz
        centers[i,:] = ir0 + w_sz//2, ic0 + w_sz//2
        win = wins[i, :, :]

        if (ir0 < 0) and (ic0 < 0):
            tmp = image[:ir1, :ic1]
            # Whiten
            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
            # subtract mean and divide by standard deviation
            wins[i, -ir0:, -ic0:] = (tmp - tmp.mean()) / adj_std
            # pad with white noise
            wins[i, :, :-ic0] = np.random.standard_normal(size=(w_sz, -ic0))
            wins[i, :-ir0, :] = np.random.standard_normal(size=(-ir0, w_sz))
        elif (ir0 < 0) and ((ic0 >= 0) and (ic1 <= im_nc)):
            tmp = image[:ir1, ic0:ic1]
            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
            wins[i, -ir0:, :] = (tmp - tmp.mean()) / adj_std
            wins[i, :-ir0, :] = np.random.standard_normal(size=(-ir0, w_sz))
        elif (ir0 < 0) and (ic1 > im_nc):
            tmp = image[:ir1, ic0:]
            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
            wins[i, -ir0:, :im_nc-ic0] = (tmp - tmp.mean()) / adj_std
            wins[i, :, im_nc-ic0:] = np.random.standard_normal(size=(w_sz, w_sz-tmp.shape[1]))
            wins[i, :-ir0, :] = np.random.standard_normal(size=(-ir0, w_sz))
        elif ((ir0 >= 0) and (ir1 <= im_nr)) and (ic0 < 0):
            tmp = image[ir0:ir1, :ic1]
            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
            wins[i, :, -ic0:] = (tmp - tmp.mean()) / adj_std
            wins[i, :, :-ic0] = np.random.standard_normal(size=(w_sz, -ic0))
        elif ((ir0 >= 0) and (ir1 <= im_nr)) and ((ic0 >= 0) and (ic1 <= im_nc)):
            tmp = image[ir0:ir1, ic0:ic1]
            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
            wins[i, :, :] = (tmp - tmp.mean()) / adj_std
        elif ((ir0 >= 0) and (ir1 <= im_nr)) and (ic1 > im_nc):
            tmp = image[ir0:ir1, ic0:]
            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
            wins[i, :, :im_nc-ic0] = (tmp - tmp.mean()) / adj_std
            wins[i, :, im_nc-ic0:] = np.random.standard_normal(size=(w_sz, w_sz-tmp.shape[1]))
        elif (ir1 > im_nr) and (ic0 < 0):
            tmp = image[ir0:, :ic1]
            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
            wins[i, :im_nr-ir0, -ic0:] = (tmp - tmp.mean()) / adj_std
            wins[i, :, :-ic0] = np.random.standard_normal(size=(w_sz, -ic0))
            wins[i, im_nr-ir0:, :] = np.random.standard_normal(size=(w_sz-tmp.shape[0], w_sz))
        elif (ir1 > im_nr) and((ic0 >= 0) and (ic1 <= im_nc)):
            tmp = image[ir0:, ic0:ic1]
            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
            win[:im_nr-ir0, :] = (tmp - tmp.mean()) / adj_std
            wins[i, im_nr-ir0:, :] = np.random.standard_normal(size=(w_sz-tmp.shape[0], w_sz))
        elif (ir1 > im_nr) and (ic1 > im_nc):
            tmp = image[ir0:, ic0:]
            adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
            wins[i, :im_nr-ir0, :im_nc-ic0] = (tmp - tmp.mean()) / adj_std
            wins[i, :, im_nc-ic0:] = np.random.standard_normal(size=(w_sz, w_sz-tmp.shape[1]))
            wins[i, im_nr-ir0:, :] = np.random.standard_normal(size=(w_sz-tmp.shape[0], w_sz))
        i += 1
            
    return wins, centers
        

def whiten(image):
    """
    """
    adjusted_std = max(image.std(), 1./ np.sqrt(image.size))
    image -= image.mean()
    
    return image/adjusted_std

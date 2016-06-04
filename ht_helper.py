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
from matplotlib import pyplot as plt
import matplotlib.patches as patches


def whiten(X):
    """
    As tensorflow.image.per_image_whitening(image)
    https://www.tensorflow.org/versions/r0.7/api_docs/python/image.html#per_image_whitening
    """
    if X.ndim == 3:
        Y = X.reshape((X.shape[0], -1))
        m = Y.mean(axis=1)
        adjusted_std = np.concatenate((Y.std(axis=1),
                                       np.repeat(1.0/np.sqrt(Y.shape[1]),
                                                 Y.shape[0])))
        adjusted_std = adjusted_std.reshape((-1, 2)).max(axis=1)   
        adjusted_std = adjusted_std.repeat(np.prod(X.shape[-2:])).reshape(X.shape)
        m = m.repeat(np.prod(X.shape[-2:])).reshape(X.shape)        
    else:
        adjusted_std = max(X.std(), 1.0/np.sqrt(X.size))
        m = X.mean()

    return (X - m) / adjusted_std


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
    ----------
    x
    y
    coordinates  - 2-d numpy.array of row and column indexes.
                   centers[:, 0] -- row indexes
                   centers[:, 1] -- columns indexes
                   Eg np.array([[r0, c0], [r1, c1], [r2, c2]])
    Returns
    -------
    c        -    the center/coordinates closest to (r, c)
    i        -    index of c in centers
    """
    
    i = np.argmin(np.abs(coordinates - [r, c]).sum(axis=1))
    c = coordinates[i, :]
    
    return c, i

class MultiResolutionPyramid:
    """
    multi resolution pyramid
    """
    def __init__(self):
        """
        """

        self.im_shape = (76, 102)
        self.N_levels = 4
        self.im_nrow = self.im_shape[0]
        self.im_ncol = self.im_shape[1]
        self.head_position = np.recarray(self.N_levels, dtype=[('x', float),('y', float)])
        self.head_position[:] = np.nan
        self.resolution = np.zeros(self.N_levels) + np.nan
        self.level_done = -1
        self.levels = []
        
        class levelClass:
            def __init__(self, nwin_r, nwin_c, win_sz):
                self.win_sz = int(win_sz)
                self.nwin_r = int(nwin_r)
                self.nwin_c = int(nwin_c)
                self.nwin = int(nwin_r * nwin_c)
                self.step = None
                self.subim_sz = None
                self.subim_pos = {'r': None, 'c': None}
                self.resolution = None
                self.valid = np.zeros(self.nwin, dtype=bool)
                self.centers = np.empty((self.nwin, 2))
                self.wins = np.empty((self.nwin, win_sz, win_sz))
                self.head_position = {'x': None, 'y': None}
 
        # Level 0
        #nwin = 5+3+5 # 1st row: 5 wins; 2nd row: 3 wins; 3rd row: 5 wins.
        nwin_r, nwin_c = 3, 4 # 3 rows x 4 columns
        win_sz = 48
        self.levels.append(levelClass(nwin_r, nwin_c, win_sz))
        self.levels[-1].step = int(win_sz/2)
        # level 1, 2 and 3
        nwin_r, nwin_c = 3, 3 # 3 rows x 4 columns
        win_sz = 32
        for _ in range(1, self.N_levels):
            self.levels.append(levelClass(nwin_r, nwin_c, win_sz))
            self.levels[-1].step = self.levels[-2].step/2
            self.levels[-1].subim_sz = self.levels[-2].step + win_sz        


    def start(self, im):
        """
        """
        if im.shape != self.im_shape:
            ValueError('Argument "im" needs to have shape (%d, %d)' % self.im_sz)
        
        self.im = im

        w_sz = self.levels[0].win_sz
        nwin_c = self.levels[0].nwin_c
        nwin_r = self.levels[0].nwin_r
        step = self.levels[0].step
        im_nr, im_nc = self.im_nrow, self.im_ncol

#        offset_c = -int(w_sz*(np.ceil(im_nc/w_sz)-(im_nc/w_sz))/2)
#        offset_r = -int(w_sz*(np.ceil(im_nr/w_sz)-(im_nr/w_sz))/2)

#        im_r0 = [offset_r, offset_r, offset_r, offset_r, offset_r,  # 1st row
#                 offset_r+step, offset_r+step, offset_r+step,       # 2nd row
#                 offset_r+2*step, offset_r+2*step, offset_r+2*step, offset_r+2*step, offset_r+2*step] # 3rd and last row
#             
#        im_c0 = [offset_c, offset_c+step, offset_c+2*step, offset_c+3*step, offset_c+4*step,  # cols in 1st row
#                 offset_c, offset_c+2*step, offset_c+4*step,                  # cols in 2nd row
#                 offset_c, offset_c+step, offset_c+2*step, offset_c+3*step, offset_c+4*step]  # cols in 3rd row 

        offset_c = int(0.5*(im_nc - (nwin_c+1)*w_sz/2))
        offset_r = int(0.5*(im_nr - (nwin_r+1)*w_sz/2))

        im_r0 = [offset_r] * nwin_c # 1st row
        im_r0.extend([offset_r + step] * nwin_c) # 2nd row
        im_r0.extend([offset_r + 2 * step] * nwin_c) # 2nd row
        
        im_c0 = [offset_c, offset_c + step,
                 offset_c + 2 * step, offset_c + 3 * step] * nwin_r
 
        # Clear/reset previous data when the object is recycled.
        self.level_done = -1
        self.head_position[:] = np.nan
        self.resolution[:] = np.nan
        for level in self.levels:
            level.valid[:] = False
            level.head_position['x'] = None
            level.head_position['y'] = None
            level.resolution = -1
            
        self._extract_and_whiten(im_r0, im_c0)
        
        self.levels[0].resolution = step
        self.resolution[0] = step
        self.level_done = 0


    def next(self, head_position):
        """
        """
        if self.level_done == self.N_levels - 1:
            print('Max resolution reached.\nBest head position estimate is '
                  'x = %1.1f and y = %1.1f.' % (self.head_position.x[-2],
                                                self.head_position.y[-2]))
            return 0
            
        if ((head_position[1] < 0) or (head_position[1] > self.im_nrow) or 
            (head_position[0] < 0) or (head_position[0] > self.im_ncol)):
            ValueError('Argument "center" needs to be within shape of "im".')
        
        if self.level_done < 0:
            print('start() needs to be run before.')
            return 0
            
        k = self.level_done + 1
        subim_sz = self.levels[k].subim_sz
        step = self.levels[k].step
        self.levels[self.level_done].head_position['x'] = head_position[1]
        self.levels[self.level_done].head_position['y'] = head_position[0]
        self.head_position[self.level_done].x = head_position[1]
        self.head_position[self.level_done].y = head_position[0]
        simr = int(round(head_position[0] - subim_sz/2))
        simc = int(round(head_position[1] - subim_sz/2))
        self.levels[self.level_done + 1].subim_pos['r'] = simr
        self.levels[self.level_done + 1].subim_pos['c'] = simc
        
        im_r0 = np.array([simr, simr, simr,
                          step+simr, step+simr, step+simr,
                          2*step+simr, 2*step+simr, 2*step+simr])

        im_c0 = np.array([simc, step+simc, 2*step+simc,
                          simc, step+simc, 2*step+simc,
                          simc, step+simc, 2*step+simc])

        self._extract_and_whiten(im_r0, im_c0)
        self.levels[k].resolution = step
        self.resolution[k] = step
        self.level_done += 1


    def _extract_and_whiten(self, im_r0, im_c0):
        """
        """
        k = self.level_done + 1
        im_nr, im_nc = self.im_nrow, self.im_ncol
        w_sz = self.levels[k].win_sz
        centers = self.levels[k].centers
        wins = self.levels[k].wins
        
        i = 0
        for ir0, ic0 in zip(im_r0, im_c0):
    
            ir1 = ir0 + w_sz            
            ic1 = ic0 + w_sz
            centers[i,:] = ir0 + w_sz//2, ic0 + w_sz//2
            win = wins[i, :, :]
    
            if (ir0 < 0) and (ic0 < 0):
                tmp = self.im[:ir1, :ic1]
                # Whiten
                adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))  # std
                # subtract mean and divide by standard deviation
                wins[i, -ir0:, -ic0:] = (tmp - tmp.mean()) / adj_std
                # pad with white noise
                wins[i, :, :-ic0] = np.random.standard_normal(size=(w_sz, -ic0))
                wins[i, :-ir0, :] = np.random.standard_normal(size=(-ir0, w_sz))
            elif (ir0 < 0) and ((ic0 >= 0) and (ic1 <= im_nc)):
                tmp = self.im[:ir1, ic0:ic1]
                adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
                wins[i, -ir0:, :] = (tmp - tmp.mean()) / adj_std
                wins[i, :-ir0, :] = np.random.standard_normal(size=(-ir0, w_sz))
            elif (ir0 < 0) and (ic1 > im_nc):
                tmp = self.im[:ir1, ic0:]
                adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
                wins[i, -ir0:, :im_nc-ic0] = (tmp - tmp.mean()) / adj_std
                wins[i, :, im_nc-ic0:] = np.random.standard_normal(size=(w_sz, w_sz-tmp.shape[1]))
                wins[i, :-ir0, :] = np.random.standard_normal(size=(-ir0, w_sz))
            elif ((ir0 >= 0) and (ir1 <= im_nr)) and (ic0 < 0):
                tmp = self.im[ir0:ir1, :ic1]
                adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
                wins[i, :, -ic0:] = (tmp - tmp.mean()) / adj_std
                wins[i, :, :-ic0] = np.random.standard_normal(size=(w_sz, -ic0))
            elif ((ir0 >= 0) and (ir1 <= im_nr)) and ((ic0 >= 0) and (ic1 <= im_nc)):
                tmp = self.im[ir0:ir1, ic0:ic1]
                adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
                wins[i, :, :] = (tmp - tmp.mean()) / adj_std
            elif ((ir0 >= 0) and (ir1 <= im_nr)) and (ic1 > im_nc):
                tmp = self.im[ir0:ir1, ic0:]
                adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
                wins[i, :, :im_nc-ic0] = (tmp - tmp.mean()) / adj_std
                wins[i, :, im_nc-ic0:] = np.random.standard_normal(size=(w_sz, w_sz-tmp.shape[1]))
            elif (ir1 > im_nr) and (ic0 < 0):
                tmp = self.im[ir0:, :ic1]
                adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
                wins[i, :im_nr-ir0, -ic0:] = (tmp - tmp.mean()) / adj_std
                wins[i, :, :-ic0] = np.random.standard_normal(size=(w_sz, -ic0))
                wins[i, im_nr-ir0:, :] = np.random.standard_normal(size=(w_sz-tmp.shape[0], w_sz))
            elif (ir1 > im_nr) and((ic0 >= 0) and (ic1 <= im_nc)):
                tmp = self.im[ir0:, ic0:ic1]
                adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
                win[:im_nr-ir0, :] = (tmp - tmp.mean()) / adj_std
                wins[i, im_nr-ir0:, :] = np.random.standard_normal(size=(w_sz-tmp.shape[0], w_sz))
            elif (ir1 > im_nr) and (ic1 > im_nc):
                tmp = self.im[ir0:, ic0:]
                adj_std = max(tmp.std(), 1.0 / np.sqrt(tmp.size))
                wins[i, :im_nr-ir0, :im_nc-ic0] = (tmp - tmp.mean()) / adj_std
                wins[i, :, im_nc-ic0:] = np.random.standard_normal(size=(w_sz, w_sz-tmp.shape[1]))
                wins[i, im_nr-ir0:, :] = np.random.standard_normal(size=(w_sz-tmp.shape[0], w_sz))
            i += 1
        
        # Only use windows with centers within im
        bi = (centers.prod(axis=1) >= 0) & ((im_nr-centers[:, 0])*(im_nc-centers[:, 1]) >= 0)
        self.levels[k].valid[bi] = True
        self.levels[k].valid[~ bi] = False
            
            
    def plot(self, level='all', true_hp=None, fname=None):
        """
        """
        fig = plt.figure(figsize=[12.55, 8.525])
        
        if level == 'all':
            self.levels[0].ax = fig.add_axes([0.025, 0.51, 0.465, 0.465])
            self.levels[1].ax = fig.add_axes([0.51, 0.51, 0.465, 0.465])
            self.levels[2].ax = fig.add_axes([0.025, 0.025, 0.465, 0.465])
            self.levels[3].ax = fig.add_axes([0.51, 0.025, 0.465, 0.465])

            for level in range(self.N_levels):
                self._plot_level(level, true_hp=true_hp)
                
        else:
            try:
                level = int(level)
                self.levels[level].ax = fig.add_subplot(111)
                self._plot_level(level, true_hp=true_hp)
            except:
                ValueError('Argument "level" must be either the string "all"'
                           'or 0 to 3 as int, string or float.')
        
        if not fname is None:
            fig.savefig(fname)
            plt.close(fig)
    
    def _plot_level(self, level, true_hp=None):


        im_nrow, im_ncol = self.im_nrow, self.im_ncol
        ax = self.levels[level].ax
        w_sz = self.levels[level].win_sz
        step = self.levels[level].step
        valid = self.levels[level].valid
        centers = self.levels[level].centers[valid]
        hp = self.levels[level].head_position
        si_sz = self.levels[level].subim_sz
        si_r = self.levels[level].subim_pos['r']
        si_c = self.levels[level].subim_pos['c']  
        cm = plt.cm.hot.from_list('jet', ['r', 'b'], N=centers.shape[0])
        
        if level == 0:
            ax.imshow(self.im, cmap=plt.cm.gray, origin='lower',
                      extent=[0, im_ncol, 0, im_nrow])
            shift = [-1, 0, 1]*5
        else:
            r0, r1 = int(max(0, si_r - 5)), int(min(im_nrow, si_r + si_sz + 5))
            c0, c1 = int(max(0, si_c - 5)), int(min(im_ncol, si_c + si_sz + 5))
            ax.imshow(self.im[r0:r1, c0:c1], cmap=plt.cm.gray,
                      origin='lower', extent=[c0, c1, r0, r1])
            shift = [-si_sz/100, si_sz/100]*5

        if not true_hp is None:
            ax.plot(true_hp['x'], true_hp['y'], marker='x', ms=14,
                    mew=2, color=[0.9, 0.5, 0.2])            

        i = 0
        for c_r, c_c in centers:
            x0, y0 = c_c - w_sz/2 + shift[i], c_r - w_sz/2 + shift[i]
            ax.add_patch(patches.Rectangle((x0, y0), w_sz, w_sz,
                                           fill=False, ls='dotted', ec=cm(i)))
            ax.plot(c_c, c_r, 'o', c=cm(i))
            i += 1

        if not hp['x'] is None:
            i = np.argmin(np.abs(centers - [[hp['y'],hp['x']]]).sum(axis=1))
            ax.plot(hp['x'], hp['y'], marker='o', ms=10, mfc='none',
                    mew='2', mec=[0.5, 1, 0.1])
            x0, y0 = hp['x'] - w_sz/2, hp['y'] - w_sz/2
            ax.add_patch(patches.Rectangle((x0, y0), w_sz, w_sz,
                                           fill=False, ls='dashed', ec=cm(i)))

        ax.set_xticks([])
        ax.set_yticks([])
        if level > 0:
            ax.set_ylim(si_r - 1, si_r + si_sz + 1)
            ax.set_xlim(si_c - 1, si_c + si_sz + 1)
        else:
            ylim = centers[:, 0].min() - step - 3, centers[:, 0].max() + step + 3
            xlim = centers[:, 1].min() - step - 3, centers[:, 1].max() + step + 3
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    
def window_and_whiten(im, window_shape, offset=(0, 0)):
    """
    Whitens the windows individually before padding with white noise.
    Whiten -- subtract mean and divide by standard deviation.

    Parameters
    ----------
    im           : 2-d numpy.array to window.
    window_shape : window shape (num_rows, num_columns)
    offset       : [row, column] list offsets where to place the upper left
                   corner of the first window.
                   The other windows follow this offset.

    Returns
    -------
    wins         : 3-d array of windows with shape
                   (num_windows, window_shape[0], window_shape[1])
    centers      : 2-array of row & column centers of the windows in the
                   original array im.
    
    Hjalmar K Turesson, 2016-05-12
    """
    im_nr, im_nc = im.shape  # number of rows, columns in image
    w_h, w_w = window_shape  # window height, width
    # Number of rows columns in "win"
    w_nr, w_nc = np.ceil(im_nr / w_h) * w_h, np.ceil(im_nc / w_w) * w_w
    
    #print('before:',offset)
    # Adjust offset
    if ((im_nr - offset[0]) / w_h) < (np.ceil(im_nr / w_h) - 1):
        offset[0] -= w_h
    if ((im_nc - offset[1]) / w_w) < (np.ceil(im_nc / w_w) - 1):
        offset[1] -= w_w
    #print('after:', offset)
    # Row indeci of windows in im. Start indeci.
    im_r0 = np.arange(offset[0], min(im_nr + offset[0], im_nr), w_h)
    im_r0[0] = max(im_r0[0], 0)  # 1st index cannot be < 0, ie outside im.
    # End indeci.
    im_r1 = np.arange(im_r0[1], im_r0[-1] + w_h + 1, w_h)
    im_r1[-1] = min(im_r1[-1], im_nr)  # Last index cannot be > num rows, ie outside im.
    # Column indeci of windows in im.
    im_c0 = np.arange(offset[1], min(im_nc+offset[1], im_nc), w_w)
    im_c0[0] = max(im_c0[0], 0)
    im_c1 = np.arange(im_c0[1], im_c0[-1] + w_w + 1, w_w)
    im_c1[-1] = min(im_c1[-1], im_nc)
    # Row indeci of windows in wins (2-d array that later will be reshaped
    # to the appropriate 3-d output array).
    win_r0 = im_r0 - offset[0]  # Row start indecei of windows.
    win_r1 = im_r1 - offset[0]  # Row end indecei of windoes.
    win_r1[-1] = min(win_r1[-1], w_nr)  # Lastt index cannot be outside of wins
    # Column indeci of windows in wins.
    win_c0 = im_c0 - offset[1]    
    win_c1 = im_c1 - offset[1]
    win_c1[-1] = min(win_c1[-1], w_nc)
    
    #print('win_c0:', win_c0)
    #print('win_c1:', win_c1)
    #print('im_c0:', im_c0)
    #print('im_c1:', im_c1)

    # 2-d array to hold the whitened windows, will later be reshaped to
    # the output 3-d array
    wins = np.random.standard_normal(size=(w_nr, w_nc))
    # Whiten each window individually
    for i in range(im_r0.shape[0]):
        for j in range(im_c0.shape[0]):
            try:
                win = im[im_r0[i] : im_r1[i], im_c0[j] : im_c1[j]]  # Extract window
                adjusted_std = max(win.std(), 1.0 / np.sqrt(win.size))  # std
                # subtract mean and divide by standard deviation
                wins[win_r0[i]: win_r1[i], win_c0[j]: win_c1[j]] = (win - win.mean()) / adjusted_std
            except:
                import pdb
                pdb.set_trace()

    # Reshape wins to 3-d, 1st step
    try:
        wins = wins.reshape(w_nr/w_h, w_h, w_nc/w_w, w_w).transpose(2, 0, 1, 3)
    except:
        import pdb
        pdb.set_trace()
    # Locations of window centers in the original array.
    # rows, columns
    n, m = wins.shape[:2]
    centers = np.zeros((m * n, 2), dtype=int)
    cr0 = offset[0] + w_h // 2
    cr1 = cr0 + (m - 1) * w_h
    centers[:, 0] = np.tile(np.arange(cr0, cr1+1, w_h), n)
    cc0 = offset[1] + w_w // 2
    cc1 = cc0 + (n - 1) * w_w
    centers[:, 1] = np.repeat(np.arange(cc0, cc1+1, w_w), m, axis=0)

    return wins.reshape((-1, w_h, w_w)), centers  # Reshape, last step


def window_image(A, window_shape, offset=(0, 0), end='cut', padvalue=0):
    """
    OBS!
    DOES NOT WHITEN THE WINDOWS BEFORE PADDING W. WHITE NOISE!
    
    Returns adjacent, non-overlapping, windows of an 2-d array.
    For overlapping windows, call window_image repeatedly with different
    offsets.

    Parameters
    ----------
    A            : 2-d numpy.array to window.
    window_shape : window shape (num_rows, num_columns)
    offset       : (row, column) offsets where to place the upper left corner
                   of the first window. The other windows follow this offset.
    end          : "cut" or "pad"
    padvalue     : A number for constant padding,
                   or the string "white" for white noise padding.
    Returns:
    --------
    B           : 3-d array of windows with shape
                  (num_windows, window_shape[0], window_shape[1])
    centers     : 2-array of row & column centers of the windows in the
                  original array A.
    
    Hjalmar K. Turesson, 2016-03-12
    """
    wr, wc = window_shape
    r0, c0 = offset
    
    if end == 'cut':

        if (r0 < 0) or (c0 < 0):
            raise ValueError("If end=='cut', then offsets have to be greater "
                             "of equal to zero")
        m, n = A[r0:, c0:].shape
        r1 = r0 + wr * (m // wr)
        c1 = c0 + wc * (n // wc)
        
    elif end == 'pad':

        m, n = A.shape
        top_pad = - min(r0, 0)
        lft_pad = - min(c0, 0)
        r0 = max(offset[0], 0)
        c0 = max(offset[1], 0)
        r1 = int(r0 + wr * np.ceil((m - r0) / wr))
        c1 = int(c0 + wc * np.ceil((n - c0) / wc))
        btm_pad = max(r1 - m, 0)
        rgt_pad = max(c1 - n, 0)
        
        if padvalue == 'white':
            pv = 0.0  # dummy, white noise is set below.
        else:
            pv = padvalue

        A = np.pad(A, ((top_pad, btm_pad), (lft_pad, rgt_pad)),
                   mode='constant',
                   constant_values=pv)
        
        if padvalue == 'white':     # White noise
            
            m, n = A.shape
            if top_pad:
                r0 = 0
                A[:top_pad, :] = np.random.standard_normal(size=(top_pad, n))
            if btm_pad:
                A[-btm_pad:, :] = np.random.standard_normal(size=(btm_pad, n))
            if lft_pad:
                c0 = 0
                A[:, :lft_pad] = np.random.standard_normal(size=(m, lft_pad))
            if rgt_pad:
                A[:, -rgt_pad:] = np.random.standard_normal(size=(m, rgt_pad))

    m, n = A[r0: r1, c0: c1].shape
    A = A[r0: r1, c0: c1].reshape(m/wr, wr, n/wc, wc).transpose(2, 0, 1, 3)
    
    # Locations of window centers in the original array.
    # rows, columns
    n, m = A.shape[:2]
    centers = np.zeros((m * n, 2), dtype=int)
    cr0 = offset[0] + wr // 2
    cr1 = cr0 + (m - 1) * wr
    centers[:, 0] = np.tile(np.arange(cr0, cr1+1, wr), n)
    cc0 = offset[1] + wc // 2
    cc1 = cc0 + (n - 1) * wc
    centers[:, 1] = np.repeat(np.arange(cc0, cc1+1, wc), m, axis=0)

    return A.reshape((-1, wr, wc)), centers

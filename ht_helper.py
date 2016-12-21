# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 18:46:36 2016

@author: hjalmar
"""
import numpy as np
import sys
from video_tools import VideoReader
from scipy.interpolate import UnivariateSpline
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import roc_auc_score, matthews_corrcoef, brier_score_loss

import ipdb


def circmedian(angs, units='deg'):
    """
    From: https://github.com/scipy/scipy/issues/6644
    """    
    
    if units == 'deg':
        angs = np.deg2rad(angs)
        
    pdists = angs[np.newaxis, :] - angs[:, np.newaxis]
    pdists = (pdists + np.pi) % (2 * np.pi) - np.pi
    pdists = np.abs(pdists).sum(1)
    
    if units == 'deg':
        return np.rad2deg(angs[np.argmin(pdists)])
    else:
        return angs[np.argmin(pdists)]


def get_error_OLD(est_track, true_track):
    """
    """
    
    if est_track.ndim > 1:
        true_track = true_track.reshape((true_track.shape[0],1))
    
    error = np.recarray(shape=est_track.shape,
                        dtype=[('position', float),
                               ('orientation', float),
                               ('orientation_weighted', float)])
    
    # Position error
    pos_err = (true_track.x - est_track.x)**2 + (true_track.y - est_track.y)**2
    error.position = np.sqrt(pos_err)
    
    # Orientation error
    error.orientation = anglediff(true_track.angle, est_track.angle, units='deg')    
    error.orientation_weighted = anglediff(true_track.angle, est_track.angle_w, units='deg')
    # no angle
    true_angle_not_ok = np.isnan(true_track.angle)
    est_angle_not_ok = np.isnan(est_track.angle)
    agree = np.logical_and(true_angle_not_ok, est_angle_not_ok)
    disagree = np.logical_xor(true_angle_not_ok, est_angle_not_ok)
    error.orientation[agree] = 0.  # if both true orientation and predicted orientation not ok -> error zero
    error.orientation_weighted[agree] = 0.
    error.orientation[disagree] = 180. # a missclassification of whether head orientation is discernible is considered 180 degree error.
    error.orientation_weighted[disagree] = 180. 

    return error     

    
def get_error(est_track, true_track):
    """
    """
    
    if est_track.ndim > 1:
        true_track = true_track.reshape((true_track.shape[0],1))
    
    error = np.recarray(shape=est_track.shape,
                        dtype=[('position', float),
                               ('orientation', float),
                               ('orientation_weighted', float)])
    
    # Position error
    pos_err = (true_track.x - est_track.x)**2 + (true_track.y - est_track.y)**2
    error.position = np.sqrt(pos_err)
    
    # Orientation error
    error.orientation = anglediff(true_track.angle, est_track.angle, units='deg')    
    error.orientation_weighted = anglediff(true_track.angle, est_track.angle_w, units='deg')
    
    descr = {}
    bix = np.logical_not(np.isnan(error.orientation))
    descr['orientation_median'] = np.median(np.abs(error.orientation[bix]))
    descr['orientation_mean'] = np.mean(np.abs(error.orientation[bix]))
    bix = np.logical_not(np.isnan(error.orientation_weighted))
    descr['orientation_weighted_median'] = np.nanmedian(np.abs(error.orientation_weighted[bix]))
    descr['orientation_weighted_mean'] = np.nanmean(np.abs(error.orientation_weighted[bix]))
    # no angle
    true_no_angle = np.isnan(true_track.angle)
    est_no_angle = np.isnan(est_track.angle)
    agree = np.logical_and(true_no_angle, est_no_angle)
    disagree = np.logical_xor(true_no_angle, est_no_angle)
    both = np.logical_or(true_no_angle, est_no_angle)
    #ipdb.set_trace()
    descr['no_angle_auc'] = roc_auc_score(true_no_angle, est_no_angle)
    descr['no_angle_mcc'] = matthews_corrcoef(true_no_angle, est_no_angle)
    descr['no_angle_brier'] = brier_score_loss(true_no_angle, est_no_angle)    
    descr['no_angle_acc'] = agree.sum()/both.sum()
    descr['no_angle_p_per_frame'] = disagree.sum()/disagree.shape[0]
    descr['position_median'] = np.median(error.position)
    descr['position_mean'] = np.mean(error.position)
    
    #print('True frequency of angle-does-not-apply:',
     #     true_no_angle.sum()/true_no_angle.shape[0])
    
    #print('Estimated frequency of angle-does-not-apply:',
     #     est_no_angle.sum()/est_no_angle.shape[0])    

    return error, descr
    

def contiguous_regions(b, minlen=1):
    """
    Finds contiguous True regions of the boolean array "b". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.

    Parameters
    ----------
    b       - an array of booleans
    minlen  - minimum length of contigous region to accept

    Returns
    -------
    idx     - 2D array where the first column is the start index of the region
              and the second column is the end index

    From Stackoverflow:
        http://stackoverflow.com/questions/4494404/
        find-large-number-of-consecutive-values-fulfilling-
        condition-in-a-numpy-array
    """
    # Find the indicies of changes in "b"
    d = np.diff(b)
    idx, = d.nonzero()
    # We need to start things after the change in "b".
    # Therefore, we'll shift the index by 1 to the right.
    idx += 1
    # If the start of condition is True prepend a 0
    if b[0]:
        idx = np.r_[0, idx]
    # If the end of condition is True, append the length of the array
    if b[-1]:
        idx = np.r_[idx, b.size]
    # Reshape the result into two columns
    idx.shape = (-1, 2)
    # Remove indeci for contigous regions shorter than minlen
    bix = (np.diff(idx) >= (minlen - 1)).flatten()
    # shift the end indecei to refer to the last position in the regions
    # otherwise the region length for regions what run until the end of "b" 
    # would be counted as 1 element shorter than those fully within "b"
    idx[:, 1] -= 1
    return idx[bix]    
    
    
def smooth(y, sigma, axis=-1, interpolation='spline'):
    """
    Does spline interpolation of missing values (NaNs) before gaussian smoothing.
    """
    
    if axis == -1:
        axis = y.ndim - 1
    elif axis == 0 or axis == 1:
        pass
    else:
        raise ValueError('axis has to be 0, 1 or -1')
        
    y = y.copy()
    x = np.arange(y.shape[axis])    

    if y.ndim == 1:
        w = np.isnan(y)
    
        if w.any():
            
            if interpolation == 'spline':
                y[w] = 0.
                spl = UnivariateSpline(x, y, w=np.logical_not(w), k=3)
                y[w] = spl(x[w])
            elif interpolation == 'linear':
                cregs = contiguous_regions(w, minlen=0)
                for cr in cregs:
                    if cr[0] > 0:
                        y0 = y[cr[0]-1]
                        
                        if cr[1] < y.shape[axis]-1:
                            y1 = y[cr[1]+1]
                            ynew = np.linspace(y0, y1, cr[1]+1-cr[0]+2, endpoint=True)
                            y[cr[0]: cr[1]+1] = ynew[1:-1]                            

                        else:  # cr[1] is last value
                            y[cr[0]:] = y0                            
                    else: # cr[0] is first value
                        y[:cr[1]+1] = y[cr[1]+1]

    elif y.ndim == 2:
        
        if axis == 0:
            y = y.T
                    
        for i in range(y.shape[0]):
            w = np.isnan(y[i])
            if w.any():
                if interpolation == 'spline':
                    y[i, w] = 0.
                    spl = UnivariateSpline(x, y[i], w=(np.logical_not(w)).astype(int), k=3)
                    y[i, w] = spl(x[w])
                elif interpolation == 'linear':
                    cregs = contiguous_regions(w, minlen=0)
                    for cr in cregs:
                        if cr[0] > 0:
                            y0 = y[i, cr[0]-1]
                            
                            if cr[1] < y.shape[1]-1:
                                y1 = y[i, cr[1]+1]
                                ynew = np.linspace(y0, y1, cr[1]+1-cr[0]+2, endpoint=True)
                                y[i, cr[0]: cr[1]+1] = ynew[1:-1]
                            else:  # cr[1] is last value
                                y[i, cr[0]:] = y0                            
                        else: # cr[0] is first value
                            y[i, :cr[1]+1] = y[i, cr[1]+1]

    else:
        raise ValueError('Only 1 or 2 dimensional input arrays are supported.')
                                                    
    return gaussian_filter1d(y, sigma, axis=axis)


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
    

class CountdownPrinter:
    
    def __init__(self, N):
        self.N = int(N)
        self.ndigits = len(str(N))
        self.fmt = '%' + '0%dd' % self.ndigits
        self.clear = '\b'*self.ndigits
    def print(self, i):
        sys.stdout.flush()
        sys.stdout.write(self.fmt % (self.N - i))
        sys.stdout.flush()
        sys.stdout.write(self.clear)


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
        
        if self.frame is None:
            return False
        
        else:
            
            self.t = self.vr.get_current_position(fmt='time')
            self.n = self.vr.get_current_position(fmt='frame_number')
            if self.n is None:
                self.n = self.t*self.vr.fps
                
            return True

    def read_frame(self, frame_num):
        """
        Read frame number frame_num. frame_num has to be >= to 0,
        and <= the total number of frames in video.
        """
        self.read_t(frame_num * self.dt)

    def next(self):
        """
        Reads next frame.
        Cannot be last frame.
        """
        self.frame = self.vr.get_next_frame()
        if self.frame is None:
            return False
        else:
            self.t = self.vr.get_current_position(fmt='time')
            self.n = self.vr.get_current_position(fmt='frame_number')
    
            if self.n is None:
                self.n = self.t*self.vr.fps
                
            return True

    def previous(self):
        """
        Reads previous frame.
        Current frame cannot the first (i.e. t = 0 and n = 1).
        """
        if (self.t - self.dt < 0) or (self.n == 1):
            print('Frame NOT updated!\n'
                  'The current frame is already the first.\n'
                  'Previous frame does not exist.')
            return False
            
        else:
            self.frame = self.vr.get_frame(self.t - self.dt)
            self.t = self.vr.get_current_position(fmt='time')
            self.n = self.vr.get_current_position(fmt='frame_number')
            if self.n is None:
                self.n = self.t*self.vr.fps
                
            return True

    def jump_t(self, jump_dur):
        """
        Jumps some duration of time from current time.
        The jump duration plus the current time has to be less than the
        total video duration.
        """
        t1 = self.vr.get_current_position(fmt='time')
        self.frame = self.vr.get_frame(t1 + jump_dur)
        
        if self.frame is None:

            return False
            
        else:            
            self.t = self.vr.get_current_position(fmt='time')
            self.n = self.vr.get_current_position(fmt='frame_number')
            if self.n is None:
                self.n = self.t*self.vr.fps
            
            return True
        

    def jump_nf(self, nframes):
        """
        Jump some number of frames from current frame.
        The number of frames to jump plus the current frame number needs to be
        less than the total number of frames in the video.
        """
        jump_dur = nframes*self.dt
        return self.jump_t(jump_dur)

    def close(self):
        """
        Close the video.
        """
        self.vr.close()


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
    
    # Chose the shortest radius since the longest will go outside of im
    if np.cos(angle) == 0:
        r = dy
    elif np.sin(angle) == 0:
        r = dx
    else:
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
        
    
def angles2complex(angles, units='deg'):
    """ 
    Angles in degrees, array or scalar
    """ 
    if units == 'deg':
        z = np.cos(np.deg2rad(angles)) + np.sin(np.deg2rad(angles))*1j
    elif units == 'rad':
        z = np.cos(angles) + np.sin(angles)*1j
    else:
        raise ValueError('"units" has to be "rad" or "deg"') 
        
    return z
    

def complex2angles(z, units='deg'):   
    """ 
    z complex array or scalar 
    angles in degrees, array or scalar
    """ 
    if units == 'deg':    
        angles = np.rad2deg(np.arctan2(z.imag, z.real))
    elif units == 'rad':
        angles = np.arctan2(z.imag, z.real)
    else:
        raise ValueError('"units" has to be "rad" or "deg"')         
        
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
            if angles_ok is None:
                y = Nclass - 1
        else:
            y[y == Nclass] = 0
            # No clear head orientation in the horiz plane,
            # ie angle_ok = 0, is coded as the last class.
            y[np.logical_not(angles_ok)] = Nclass - 1
        
    return np.float32(y)
    
    
def circmean_weighted(angles, w, axis=0, units='deg'):
    
    if units == 'deg':
        angles = np.deg2rad(angles)
    elif units == 'rad':
        pass
    else:
        raise ValueError('"units" has to be "rad" or "deg"')
        
    s = np.nansum(np.sin(angles) * w, axis=axis)
    c = np.nansum(np.cos(angles) * w, axis=axis)
    
    wmean = np.arctan2(s, c)
    
    if units == 'deg':
        wmean = np.rad2deg(wmean)
    
    return wmean
    
    
def class2angle(y,  Nclass, units='deg'):
    """
    Angles are from -180 to 180
    y run from 0 to Nclass-1
    """
    if units == 'deg':
        angles = y * (360 / Nclass) + 180 / Nclass # angles run from -180:180
    elif units == 'rad':
        angles = y * (np.pi / Nclass) + np.pi / (2 * Nclass) # angles run from -180:180
    else:
        raise ValueError('"units" has to be "rad" or "deg"')         
        
    return angles - 180
    
   
def whiten(image):
    """
    """
    adjusted_std = max(image.std(), 1./ np.sqrt(image.size))
    image -= image.mean()
    
    return image/adjusted_std

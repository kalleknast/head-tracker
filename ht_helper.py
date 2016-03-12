# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 18:46:36 2016

@author: hjalmar
"""
import termios
import fcntl
import sys
import os
import numpy as np


def whiten(X):
    """
    As tensorflow.image.per_image_whitening(image)
    https://www.tensorflow.org/versions/r0.7/api_docs/python/image.html#per_image_whitening
    """
    adjusted_std = max(X.std(), 1.0/np.sqrt(X.size))
    return (X - X.mean()) / adjusted_std


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


def window_image(A, window_shape, offset=(0, 0), end='cut', padvalue=0):
    """
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
    wr, wc = window_shape[0], window_shape[1]    
    r0, c0 = offset[0], offset[1]   
    
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
        r1 = int(r0 + wr * np.ceil(m / wr))
        c1 = int(c0 + wc * np.ceil(n / wc))        
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

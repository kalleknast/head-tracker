# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:33:25 2016

@author: hjalmar
"""


from ht_helper import get_input, whiten, window_and_whiten, FrameStepper
from ht_helper import MultiResolutionPyramid, closest_coordinate
from matplotlib import pyplot as plt
from datetime import datetime
from scipy.misc import imresize
from scipy.spatial.distance import pdist, squareform
from time import sleep
from glob import glob
import numpy as np
import warnings
import tables
import sys

warnings.simplefilter("ignore")


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
                  self.video_fname.split('/')[-1],
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
        print('"units" has to be "rad" or "deg"')
        return 0
    
    return gzl_x, gzl_y


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

    # Get position of box around head.
    bx_wdt, bx_lngt = data['box_width'], data['box_height']
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


def batch_train_data_to_storage(log_dir, video_dir, out_fname,
                                max_patch_overlap=0.1, num_neg_per_pos=11):
    """
    Loads training data from all the log files in a directory.
    """
                    
    
    log_fnames = glob(log_dir.rstrip('/') + '/HeadData*.txt')

    log_fname = log_fnames.pop(0)
    train_data_to_storage(log_fname, video_dir, out_fname, mode='w',
                          max_patch_overlap=max_patch_overlap,
                          num_neg_per_pos=num_neg_per_pos)
    for log_fname in log_fnames:
        print(log_fname)
        train_data_to_storage(log_fname, video_dir, out_fname, mode='a',
                              max_patch_overlap=max_patch_overlap,
                              num_neg_per_pos=num_neg_per_pos)
                              
    f = tables.openFile(out_fname, mode='r')
    print('Number of positive exemplars entered: %d' 
          % (f.root.labels[:,0]==1).sum())
    f.close()


def traindata2storage_centeredgrid(log_fname, video_dir, out_fname, mode='a',
                                   head_size=200, patch_size=32,
                                   patch_overlap=0.5, plot=False, plot_dir='.'):
    """
    Parameters
    ----------
    log_fname
    video_dir
    out_fname
    head_size
    patch_size
    patch_overlap
    plot
    plot_dir
    """
    
    # TODO: remove hard coded frame sizes
    # number of patches over the width of frame
    n_patch_w = int(np.ceil(640 / head_size))
    # number of patches over the height of frame
    n_patch_h = int(np.ceil(480 / head_size))

    scale_factor = patch_size / head_size

    log_data, log_header = read_log_data(log_fname)
    video_fname = '%s/%s' % (video_dir.rstrip('/'), log_header['video_fname'])
    video_fname = glob(video_fname)
    if len(video_fname) != 1:
        print('The video file matching to the log cannot be found in %s'
              % video_dir)
        print('Log file:', log_fname)
        print('Found video files:', video_fname)
        return 0
    log_fname = log_fname.split('/')[-1]
    fstp = FrameStepper(video_fname[0])
    
    if plot:
        pltdat = {'dir': plot_dir.rstrip('/'), 'poverlap': patch_overlap,
                  'np_w': n_patch_w, 'np_h': n_patch_h, 'p_sz': patch_size}
        pltdat = plt_traindata2storage_grid(pltdat, ptype='prepare')

    # HDF5 storage of the data
    f = tables.openFile(out_fname, mode=mode)
    if mode == 'w':  # Write new file -- OVERWRITES!
        atom_data = tables.Atom.from_dtype(fstp.frame.dtype)
        atom_labels = tables.Atom.from_dtype(np.dtype(int))
        filters = tables.Filters(complib='blosc', complevel=5)

        class tbl_spec(tables.IsDescription):
            frm_n   = tables.IntCol(pos=1)        # int, frame number in video
            log_fn  = tables.StringCol(50, pos=2) # 50-character str 
                                                  # (48 seems to be max)


        data_origin = f.create_table(f.root, 'data_origin', tbl_spec,
                                     "Table with info about data origin")
                                
        data = f.createEArray(f.root, 'data', atom_data,
                              shape=(0, patch_size, patch_size),
                              filters=filters,
                              chunkshape=(patch_size, patch_size, 1))
        labels = f.createEArray(f.root, 'labels', atom_labels,
                                shape=(0, 2),
                                filters=filters,
                                chunkshape=(1, 2))
    elif mode == 'a':
        # Open existing file to append data.
        data = f.root.data
        labels = f.root.labels
        data_origin = f.root.data_origin

    # Scale head positions to fit rescaled frames.
    log_data['center_x'] = log_data['center_x'] * scale_factor
    log_data['center_y'] = log_data['center_y'] * scale_factor
    # Head angles to degrees (-180 to 180)
    log_data['angle'] = (180 * (log_data['angle'] / np.pi)).round()

    for i, dat in enumerate(log_data):
        # Counter printed on the command line
        s = '%04d' % (log_data.shape[0] - i)
        sys.stdout.write(s)
        sys.stdout.flush()
        sys.stdout.write('\b'*len(s))
        # Read the frame
        #frame_num = dat['frame_num']
        fstp.read_t(dat['frame_time'])
        frame = imresize(fstp.frame.mean(axis=2), scale_factor)
                
        if plot:
            plt_traindata2storage_grid(pltdat, ptype='clear')
            pltdat['frame'] = frame
            plt_traindata2storage_grid(pltdat, ptype='show_frame')

        # Location of positive example, i.e. head
        pos_x, pos_y = dat['center_x'], dat['center_y']
        # offsets for patch extraction
        offset_c = pos_x - patch_size / 2
        offset_c = int(round(min(np.mod(offset_c, patch_size), offset_c))) #- 32
        offset_r = pos_y - patch_size / 2
        offset_r = int(round(min(np.mod(offset_r, patch_size), offset_r))) #- 32
        step = int(patch_overlap * patch_size)

        for offs_r in range(offset_r, offset_r + patch_size, step):
            for offs_c in range(offset_c, offset_c + patch_size, step):
                patches, centers = window_and_whiten(frame,
                                                     (patch_size, patch_size),
                                                     [offs_r, offs_c])
                ppix = np.logical_and(abs(centers[:, 0] - pos_y) < 2,
                                      abs(centers[:, 1] - pos_x) < 2)
                npatch = patches.shape[0]
                if ppix.any():
                    pos_patch = patches[ppix, :, :]
                    # Append data to storage file.        
                    labels.append(np.array([1, dat['angle']], dtype=int).reshape((1, 2)))
                    data.append(pos_patch.reshape((1, patch_size, patch_size)))
                    data_origin.append([(dat['frame_num'], log_fname)])

                # Negative examples, i.e. background/no head
                neg_patch_ids = np.nonzero(~ ppix)[0]  
                for npix in neg_patch_ids:
                    neg_patch = patches[npix, :, :]
                    # Append data to storage file.
                    labels.append(np.array([0, 0], dtype=int).reshape((1, 2)))
                    data.append(neg_patch.reshape((1, patch_size, patch_size)))
                    data_origin.append([(dat['frame_num'], log_fname)])
            
                if plot:
                    pltdat['npatch'], pltdat['dat'] = npatch, dat
                    pltdat['patches'], pltdat['centers'] = patches, centers
                    pltdat['ppix'], pltdat['neg_patch_ids'] = ppix, neg_patch_ids
                    plt_traindata2storage_grid(pltdat, ptype='show_patches')
                #
        if plot:
            pltdat['i'] = i
            plt_traindata2storage_grid(pltdat, ptype='save_fig')
            if not get_input('Continue', bool, default=True):
                print('Quitting')
                fstp.close()
                f.close()
                return

    fstp.close()
    f.close()


def traindata2storage_grid(log_fname, video_dir, out_fname, head_size=200,
                           mode='a', patch_size=32, align_grid2head = False,
                           patch_overlap=0.5, plot=False, plot_dir='.'):
    """
    Parameters
    ----------
    log_fname
    video_dir
    out_fname
    head_size
    patch_size
    patch_overlap
    plot
    plot_dir
    """
    
    # TODO: remove hard coded frame sizes
    # number of patches over the width of frame
    n_patch_w = int(np.ceil(640 / head_size))
    # number of patches over the height of frame
    n_patch_h = int(np.ceil(480 / head_size))
    scale_factor = patch_size / head_size
    max_dist2head = np.ceil(patch_size * patch_overlap/2)

    log_data, log_header = read_log_data(log_fname)
    video_fname = '%s/%s' % (video_dir.rstrip('/'), log_header['video_fname'])
    video_fname = glob(video_fname)
    if len(video_fname) != 1:
        print('The video file matching to the log cannot be found in %s'
              % video_dir)
        print('Log file:', log_fname)
        print('Found video files:', video_fname)
        return 0
    log_fname = log_fname.split('/')[-1]
    fstp = FrameStepper(video_fname[0])
    
    if plot:
        pltdat = {'dir': plot_dir.rstrip('/'), 'poverlap': patch_overlap,
                  'np_w': n_patch_w, 'np_h': n_patch_h, 'p_sz': patch_size}
        pltdat = plt_traindata2storage_grid(pltdat, ptype='prepare')

    # HDF5 storage of the data
    f = tables.openFile(out_fname, mode=mode)
    if mode == 'w':  # Write new file -- OVERWRITES!
        atom_data = tables.Atom.from_dtype(fstp.frame.dtype)
        atom_labels = tables.Atom.from_dtype(np.dtype(int))
        filters = tables.Filters(complib='blosc', complevel=5)

        class tbl_spec(tables.IsDescription):
            frm_n   = tables.IntCol(pos=1)        # int, frame number in video
            log_fn  = tables.StringCol(50, pos=2) # 50-character str 
                                                  # (48 seems to be max)


        data_origin = f.create_table(f.root, 'data_origin', tbl_spec,
                                     "Table with info about data origin")
                                
        data = f.createEArray(f.root, 'data', atom_data,
                              shape=(0, patch_size, patch_size),
                              filters=filters,
                              chunkshape=(patch_size, patch_size, 1))
        labels = f.createEArray(f.root, 'labels', atom_labels,
                                shape=(0, 2),
                                filters=filters,
                                chunkshape=(1, 2))
    elif mode == 'a':
        # Open existing file to append data.
        data = f.root.data
        labels = f.root.labels
        data_origin = f.root.data_origin

    # Scale head positions to fit rescaled frames.
    log_data['center_x'] = log_data['center_x'] * scale_factor
    log_data['center_y'] = log_data['center_y'] * scale_factor
    # Head angles to degrees (-180 to 180)
    log_data['angle'] = (180 * (log_data['angle'] / np.pi)).round()

    for i, dat in enumerate(log_data):
        # Counter printed on the command line
        s = '%04d' % (log_data.shape[0] - i)
        sys.stdout.write(s)
        sys.stdout.flush()
        sys.stdout.write('\b'*len(s))
        # Read the frame
        #frame_num = dat['frame_num']
        fstp.read_t(dat['frame_time'])
        frame = imresize(fstp.frame.mean(axis=2), scale_factor)
                
        if plot:
            plt_traindata2storage_grid(pltdat, ptype='clear')
            pltdat['frame'] = frame
            plt_traindata2storage_grid(pltdat, ptype='show_frame')

        # Location of positive example, i.e. head
        pos_x, pos_y = dat['center_x'], dat['center_y']
        # offsets for patch extraction
        if align_grid2head:
            offset_c = pos_x - patch_size / 2
            offset_c = int(round(min(np.mod(offset_c, patch_size), offset_c)))
            offset_r = pos_y - patch_size / 2
            offset_r = int(round(min(np.mod(offset_r, patch_size), offset_r)))
        else:
            offset_c = -int((frame.shape[1]/patch_size - frame.shape[1]//patch_size) * patch_size/2)
            offset_r = -int((frame.shape[0]/patch_size - frame.shape[0]//patch_size) * patch_size/2)
        #print('offset_c:',offset_c, 'offset_r:',offset_r)
        step = int(patch_overlap * patch_size)

        for offs_r in range(offset_r, offset_r + patch_size, step):
            for offs_c in range(offset_c, offset_c + patch_size, step):
                patches, centers = window_and_whiten(frame,
                                                     (patch_size, patch_size),
                                                     [offs_r, offs_c])
                ppix = np.logical_and(abs(centers[:, 0] - pos_y) < max_dist2head,
                                      abs(centers[:, 1] - pos_x) < max_dist2head)
                npatch = patches.shape[0]
                if ppix.any():
                    pos_patch = patches[ppix, :, :]
                    # Append data to storage file.        
                    labels.append(np.array([1, dat['angle']], dtype=int).reshape((1, 2)))
                    data.append(pos_patch.reshape((1, patch_size, patch_size)))
                    data_origin.append([(dat['frame_num'], log_fname)])

                # Negative examples, i.e. background/no head
                neg_patch_ids = np.nonzero(~ ppix)[0]  
                for npix in neg_patch_ids:
                    neg_patch = patches[npix, :, :]
                    # Append data to storage file.
                    labels.append(np.array([0, 0], dtype=int).reshape((1, 2)))
                    data.append(neg_patch.reshape((1, patch_size, patch_size)))
                    data_origin.append([(dat['frame_num'], log_fname)])
            
                if plot:
                    pltdat['npatch'], pltdat['dat'] = npatch, dat
                    pltdat['patches'], pltdat['centers'] = patches, centers
                    pltdat['ppix'], pltdat['neg_patch_ids'] = ppix, neg_patch_ids
                    plt_traindata2storage_grid(pltdat, ptype='show_patches')
                #
        if plot:
            pltdat['i'] = i
            plt_traindata2storage_grid(pltdat, ptype='save_fig')
            if not get_input('Continue', bool, default=True):
                print('Quitting')
                fstp.close()
                f.close()
                return

    fstp.close()
    f.close()


def labelledData2storage_mrp_batch(log_dir, video_dir, out_fname):
    """
    Loads training data from all the log files in a directory.
    
    Trying to maximize resolution while minimizing the number of windows/patches
    to classify per video fram using a Multi Resolution Pyramid.
    """
                    
    
    log_fnames = glob(log_dir.rstrip('/') + '/HeadData*.txt')

    log_fname = log_fnames.pop(0)
    print(log_fname)
    labeledData2storage_mrp(log_fname, video_dir, out_fname, mode='w')

    for log_fname in log_fnames:
        print(log_fname)
        labeledData2storage_mrp(log_fname, video_dir, out_fname, mode='a')
                              
    f = tables.openFile(out_fname, mode='r')
    Npos = (f.root.labels0[:,0]==1).sum()
    Nneg = (f.root.labels0[:,0]==0).sum()
    print('number of POSITIVE exemplars: %d\n'
          'number of NEGATIVE exemplars: %d' % (Npos, Nneg))
    f.close()


def labeledData2storage_mrp(log_fname, video_dir, out_fname,
                            mode='a', plot=False, plot_dir='.'):
    """
    Trying to maximize resolution while minimizing the number of windows/patches
    to classify per video fram using a Multi Resolution Pyramid.
    
    Parameters
    ----------
    log_fname
    video_dir
    out_fname
    head_size
    patch_size
    plot
    plot_dir
    """
    
    head_size = 200
    patch_sz = [48, 32, 32, 32]
    scale_factor = patch_sz[1] / head_size  # TODO improve this 

    log_data, log_header = read_log_data(log_fname)
    video_fname = '%s/%s' % (video_dir.rstrip('/'), log_header['video_fname'])
    video_fname = glob(video_fname)
    if len(video_fname) != 1:
        print('The video file matching to the log cannot be found in %s'
              % video_dir)
        print('Log file:', log_fname)
        print('Found video files:', video_fname)
        return 0
    log_fname = log_fname.split('/')[-1]
    # To read the frames
    fstp = FrameStepper(video_fname[0])
    # For patch extraction
    mrp = MultiResolutionPyramid()
    
    # HDF5 storage of the data
    f = tables.openFile(out_fname, mode=mode)
    if mode == 'w':  # Write new file -- OVERWRITES!
        atom_data = tables.Atom.from_dtype(fstp.frame.dtype)
        atom_labels = tables.Atom.from_dtype(np.dtype(int))
        filters = tables.Filters(complib='blosc', complevel=5)

        class tbl_spec(tables.IsDescription):
            frm_n   = tables.IntCol(pos=1)        # int, frame number in video
            log_fn  = tables.StringCol(50, pos=2) # 50-character str 
                                                  # (48 seems to be max)
        data_src, data, labels = [], [], []
        for k in range(mrp.N_levels):
            data_src_title = "Info about the source of data from Level %d" % k
            data_src.append(f.create_table(f.root, ('data%d_src' % k),
                                           tbl_spec, title=data_src_title))

            data_title = 'Image patches from Level %d of the multi-resolution pyramid' % k
            data.append(f.createEArray(f.root, ('data%d' % k), atom_data,
                                       shape=(0, patch_sz[k], patch_sz[k]),
                                       title=data_title, filters=filters,
                                       chunkshape=(patch_sz[k], patch_sz[k], 1)))

            labels_title = 'Labels for level %d image patches of the multi-resolution pyramid' % k
            labels.append(f.createEArray(f.root, ('labels%d' % k), atom_labels,
                          shape=(0, 2), title=labels_title,
                          filters=filters, chunkshape=(1, 2)))
                          
    elif mode == 'a':
        # Open existing file to append data.
        data_src, data, labels = [], [], []
        for k in range(mrp.N_levels):
            eval('data_src.append(f.root.%s)' % ('data%d_src' % k))
            eval('data.append(f.root.%s)' % ('data%d' % k))
            eval('labels.append(f.root.%s)' % ('labels%d' % k))

    # Scale head positions to fit rescaled frames.
    log_data['center_x'] = log_data['center_x'] * scale_factor
    log_data['center_y'] = log_data['center_y'] * scale_factor
    # Head angles to degrees (-180 to 180)
    log_data['angle'] = (180 * (log_data['angle'] / np.pi)).round()

    for i, dat in enumerate(log_data):
        # Counter printed on the command line
        s = '%04d' % (log_data.shape[0] - i)
        sys.stdout.write(s)
        sys.stdout.flush()
        sys.stdout.write('\b'*len(s))
        # Read the frame
        fstp.read_t(dat['frame_time'])
        frame = imresize(fstp.frame.mean(axis=2), scale_factor)
        # Location of positive example, i.e. head
        pos_x, pos_y = dat['center_x'], dat['center_y']
        mrp.start(frame)
        c, cj = closest_coordinate(pos_y, pos_x, mrp.levels[0].centers)
        for j, patch in enumerate(mrp.levels[0].wins):
            if mrp.levels[0].valid[j]:
                # Append data to storage file.
                pp = int(j == cj)  # 1 -> Positive Patch, ie w. head, 0 -> no head
                labels[0].append(np.array([pp, dat['angle']], dtype=int).reshape((1, 2)))
                data[0].append(patch.reshape((1, patch_sz[0], patch_sz[0])))
                data_src[0].append([(dat['frame_num'], log_fname)])
        
        for k in range(1, mrp.N_levels):
            mrp.next(c)
            c, cj = closest_coordinate(pos_y, pos_x, mrp.levels[k].centers)

            for j, patch in enumerate(mrp.levels[k].wins):
                if mrp.levels[k].valid[j]:                
                    # Append data to storage file.
                    pp = int(j == cj)  # 1 -> Positive Patch, ie w. head, 0 -> no head
                    labels[k].append(np.array([pp, dat['angle']], dtype=int).reshape((1, 2)))
                    data[k].append(patch.reshape((1, patch_sz[k], patch_sz[k])))
                    data_src[k].append([(dat['frame_num'], log_fname)])

        if plot:
            mrp.levels[-1].head_position['x'] = c[1]
            mrp.levels[-1].head_position['y'] = c[0]
            mrp.plot(level='all', true_hp={'x': pos_x, 'y': pos_y},
                     fname=('%s/mrp_hp%03d.png' % (plot_dir, i)))
            #if not get_input('Continue', bool, default=True):
            #if i > 4:
             #   print('Quitting')
              #  fstp.close()
               # f.close()
                #return

    fstp.close()
    f.close()


def plt_traindata2storage_grid(pltdat, ptype):
    """
    """
    
    if ptype == 'prepare':
        
        np_w = pltdat['np_w']
        np_h = pltdat['np_h']
        poverlap = pltdat['poverlap']          
        pltdat['axs'] = []
        pltdat['linecm'] = plt.cm.hot.from_list('jet', ['r', 'b'], N=(1/poverlap)**2-1)
        pltdat['fig'] = plt.figure(figsize=[11.4125, 8.525])
        # complete frame
        # upper left quarter
        pltdat['axs'].append(pltdat['fig'].add_axes([0.01, 0.51, 0.48, 0.48]))
        # patches
        sp_w, sp_h = 0.48 / np_w, 0.48 / np_h
        # upper right quarter
        for x in np.linspace(0.51, 0.99 - sp_w, np_w):
            for y in np.linspace(0.51, 0.99 - sp_h, np_h):
                pltdat['axs'].append(pltdat['fig'].add_axes([x, y, sp_w, sp_h]))
        # lower left quarter
        for x in np.linspace(0.01, 0.49-sp_w, np_w):
            for y in np.linspace(0.01, 0.49-sp_h, np_h):
                pltdat['axs'].append(pltdat['fig'].add_axes([x, y, sp_w, sp_h]))
        # lower right quarter
        for x in np.linspace(0.51, 0.99-sp_w, np_w):
            for y in np.linspace(0.01, 0.49-sp_h, np_h):
                pltdat['axs'].append(pltdat['fig'].add_axes([x, y, sp_w, sp_h]))

    elif ptype == 'clear':
        
        pltdat['k'] = 1
        
        for ax in pltdat['axs']:
            ax.cla()
            ax.set_xticks([])
            ax.set_yticks([])
            
    elif ptype == 'show_frame':
        
        pltdat['axs'][0].imshow(pltdat['frame'],
                               origin='lower',
                               cmap=plt.cm.gray)
        pltdat['axs'][0].set_xlim([0, pltdat['frame'].shape[1]])
        pltdat['axs'][0].set_ylim([0, pltdat['frame'].shape[0]])
    
    elif ptype == 'show_patches':
        
        k = pltdat['k']
        n = pltdat['npatch']
        axs = pltdat['axs']    
        ps = pltdat['patches']
        cs = pltdat['centers']
        p_sz = pltdat['p_sz']
        ppix = pltdat['ppix']
        pos_x = pltdat['dat']['center_x']
        pos_y = pltdat['dat']['center_y']
        angle = pltdat['dat']['angle']
        npids = pltdat['neg_patch_ids']

        lclr = pltdat['linecm']((k // n) / pltdat['linecm'].N)
        ls = ':'
        if ppix.any():
            axs[0].plot(cs[ppix, 1], cs[ppix, 0], 'og')
            pltdat['axs'][0].plot(pos_x, pos_y, 'o', mec='g',
                                  mew='1', ms=16, mfc='none')
            ls = '--'
    
        axs[0].plot(cs[npids, 1], cs[npids, 0], 'or')                    
        rows = np.r_[cs[:, 0] + p_sz/2, cs[:, 0] - p_sz/2]
        cols = np.r_[cs[:, 1] + p_sz/2, cs[:, 1] - p_sz/2]
        xlim, ylim = axs[0].get_xlim(), axs[0].get_ylim()

        for r in np.unique(rows):
            axs[0].plot(xlim, [r+0.33*((k // n) / pltdat['linecm'].N - 0.5),
                               r+0.33*((k // n) / pltdat['linecm'].N - 0.5)],
                        ls=ls, c=lclr, zorder=1)
        for c in np.unique(cols):
            axs[0].plot([c+(k-1)//cs.shape[0] - 1.5, 
                         c+(k-1)//cs.shape[0] - 1.5],
                         ylim, ls=ls, c=lclr, zorder=1)
                
        if (k + n <= len(axs)):
            for i in range(n):
                axs[i + k].imshow(ps[i, :, :], origin='lower',
                                  cmap=plt.cm.gray, clim=[-2, 2])
                axs[i + k].plot([0, p_sz-1,],[0,0], c=lclr, ls=ls)
                axs[i + k].plot([0, p_sz-1], [p_sz-1,p_sz-1], c=lclr, ls=ls)
                axs[i + k].plot([0, 0], [0, p_sz-1], c=lclr, ls=ls)
                axs[i + k].plot([p_sz-1, p_sz-1], [0, p_sz-1], c=lclr, ls=ls)

                if i == ppix.nonzero()[0]:  # Mark head patch and angle
                    gzl_x, gzl_y = get_gaze_line(angle, p_sz/2, p_sz/2, 0.4*p_sz, units='deg')
                    axs[i + k].plot(gzl_x, gzl_y, '-g')
                    axs[i + k].text(gzl_x[0], gzl_y[0], '%d' % angle, color='w')
                    axs[i + k].plot(p_sz/2, p_sz/2, 'og')
                                 
        pltdat['k'] = k + n
            
    elif ptype == 'save_fig':
        pltdat['fig'].savefig('%s/hp%03d.png' % (pltdat['dir'], pltdat['i']))
        
    return pltdat


def traindata2storage_biased(log_fname, video_dir, out_fname, mode='a',
                             head_size=200, patch_size=32,
                             max_patch_overlap=0.25,
                             num_neg_per_pos=2, debug=False):
    """
    Sample negative patches from ...
    
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
    # Min distance to positive sample
    min_dst_to_pos = patch_size * max_patch_overlap

    log_data, log_header = read_log_data(log_fname)
    video_fname = '%s/%s' % (video_dir.rstrip('/'), log_header['video_fname'])
    video_fname = glob(video_fname)
    if len(video_fname) != 1:
        print('The video file matching to the log cannot be found in %s'
              % video_dir)
        print('Log file:', log_fname)
        print('Found video files:', video_fname)
        return 0
    log_fname = log_fname.split('/')[-1]
    fstp = FrameStepper(video_fname[0])
    # TODO: check that below can be removed
    #frame = whiten(imresize(fstp.frame, scale_factor))

    # HDF5 storage of the data
    f = tables.openFile(out_fname, mode=mode)
    if mode == 'w':  # Write new file -- OVERWRITES!
        atom_data = tables.Atom.from_dtype(frame.dtype)
        atom_labels = tables.Atom.from_dtype(np.dtype(int))
        filters = tables.Filters(complib='blosc', complevel=5)

        class tbl_spec(tables.IsDescription):
            frm_n   = tables.IntCol(pos=1)        # int, frame number in video
            log_fn  = tables.StringCol(50, pos=2) # 50-character str 
                                                  # (48 seems to be max)


        data_origin = f.create_table(f.root, 'data_origin', tbl_spec,
                                     "Table with info about data origin")
                                
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
        data = f.root.data
        labels = f.root.labels
        data_origin = f.root.data_origin

    # Scale head positions and add padding
    log_data['center_x'] = log_data['center_x'] * scale_factor
    log_data['center_y'] = log_data['center_y'] * scale_factor
    # Head angles to degrees (-180 to 180)
    log_data['angle'] = (180 * (log_data['angle'] / np.pi)).round()

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
        
        # Append data to storage file.        
        labels.append(np.array([1, dat['angle']], dtype=int).reshape((1, 2)))
        data.append(pos_patch.reshape((1, patch_size, patch_size)))
        data_origin.append([(dat['frame_num'], log_fname)])

        # Negative examples, i.e. background/no head
        all_xy = np.zeros((num_neg_per_pos + 1, 2))
        neg_idx = np.random.randint(0, log_data.shape[0], num_neg_per_pos)
        all_xy[0, :] = pos_x, pos_y
        all_xy[1:, 0] = log_data['center_x'][neg_idx]
        all_xy[1:, 1] = log_data['center_y'][neg_idx]
        ok = pdist(all_xy, metric='euclidean') > min_dst_to_pos
        nok = squareform(~ ok).sum(axis=0)
        while nok.any():  # Check that the randomly drawn position are
                          # not within head position in current frame.
            neg_idx = np.random.randint(0, log_data.shape[0], 1)
            # Always keep the positive (pos_x, pos_y)
            all_xy[1:, 0][nok[1:].argmax()] = log_data['center_x'][neg_idx]
            all_xy[1:, 1][nok[1:].argmax()] = log_data['center_y'][neg_idx]
            ok = pdist(all_xy, metric='euclidean') > min_dst_to_pos
            nok = squareform(~ ok).sum(axis=0)           

        j = 2
        for neg_x, neg_y in all_xy[1:]:
            neg_patch = extract_patch(neg_x, neg_y, frame)
            # Append data to storage file.
            labels.append(np.array([0, 0], dtype=int).reshape((1, 2)))
            data.append(neg_patch.reshape((1, patch_size, patch_size)))
            data_origin.append([(dat['frame_num'], log_fname)])
            if debug and j < 6:
                axs[j].imshow(neg_patch, origin='lower', cmap=plt.cm.gray)
                j += 1

        if debug:
            axs[0].plot(all_xy[1:, 0], all_xy[1:, 1], 'or')
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
        # TODO: do it w reshape instead of loops.
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
# head-tracker

## What it is
Routines for video tracking of head position and orientation in the common marmoset. KNOWING POS AND ORIENT TELLS US WHERE THEY LOOK BC... SACCADES, HEAD SACCADES. That is, these routines to estimate where the subject is and roughly where it is looking using only a webcam.

## What it does

## How it works

## Example usage
- Label training data
```
from data_preparation import LabelFrames
video_fname = '/full/path/to/some/video/file.mkv'
lf = LabelFrames(video_fname)

lf.run_batch(t0=2.3)   # time of 1st frame to label in seconds
```
- Extract training data and store in tfrecords format.
```
log_dir = '/directory/where/log/files/are/stored/'
video_dir = '/directory/where/video/files/are/stored/'
out_base_fname = 'name'  # TODO names will be like this...
out_dir = '/directory/where/training/and/dev/data/will/be/stored'
labeledData2storage_mrp(log_dir, video_dir, out_base_fname, out_dir, plot=False, plot_dir='.', Nplot=500)
```
- Train the position classifier/tracker


## Further information

## In progress
- DONE read frames sequentially one-by-one from video.
- DONE display and manually annote frames.

    * the reward plate is minimum 20 degrees wide.
    * 10 degree bins would be great.
    * 10 deg bins -> 36 bins
    * 1 000 samples/bin * 36 bins -> 36 000 samples.
    * DONE current frame number and remaining frame number in figure.
    * DONE a stopping mechanism after a finished sequence.
    * DONE raw positions to log file, i.e. positions of the 3 clicks.
- DONE export the data in a good format.
    * DONE positions and angles to a text file
    * SKIP pngs with the head boxes.
- DONE Data check, i.e. plot/draw data on frames for visual check.
- DONE Frame scaling.
- A cool classifier.
- DONE image windowing for sliding windows/head localization.
- Automatically annote frames from classifier predictions.

# Issues
- Doesn't draw gaze line at pi and 3*pi/2.

# Classification strategy
1. A head detector on a window that is translated, rotated and scaled over all
   possible positions, rotations and scales. Best fit wins.
    - A bit dumb, probably too slow for real time application.
    - Should require too much training data.
    
2. Train on raw video frames (down-scaled) using head_x, head_y and angle as
   labels/classes.
    - Would probably require very much data since the number of classes becomes
      very big.

3. Separate head position detector (invariant to rotation) and 
   rotation detector (invariant to translation).
    - Since position and rotation seems to become independent, this approach 
      should result in significantly less classes than strategy 2.
    - Could still result in too many classes.

4. Only consider translations in a local neighborhood to previous frame's
   head position and rotation. Take advantage of the fact that translations
   and rotation are continous without abrupt jumps.
    - Requires and "anchor frame" to start with. I.e. a frame where
      head position and rotation is known (manually labelled). From the known
      frame the classifier will search for the follwing frame's position and
      rotation.
    - Since the neighborhood is smaller than the entire frame, the number
      of possible positions and rotations are also smaller. I.e. the number
      of labels should be significantly less than for strategy 2.

Testing strategy 3 first.


# head-tracker
Routines for video tracking of gaze direction in the common marmoset.

# What I will need

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
- Automatically annote frames from classifier predictions.

# PROBLEMS
- Doesn't draw gaze line at pi and 3*pi/2.

# Structure of Classifier code.
- Predictor
    * load weights
    * moving window over frames
    * best window selector
- Trainer
    * save weights (load for accumulative training?)
- Configure network
- Split and randomize data
    * load data


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

Strategy 2 is probably easiest to implement/test, and should thus be tried 1st.


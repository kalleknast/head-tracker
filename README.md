# head-tracker

## What it is
Current sophisticated technologies for manipulating and recording the nervous system allow us to perform historically unprecedented experiments. However, any influences of our experimental manipulations might have on psychological processes must be inferred from their effects on behavior. Today, quantifying behavior has become the bottleneck for large-scale, high throughput, experiments.

**head-tracker** try to solve this issue through using modern deep learning algorithms for video based animal tracking. It provides methods for tracking head position and orientation (i.e. indirectly gaze) from simple video recordings of the common marmoset (*Callithrix jacchus*).

## What it does

Given a video of a marmoset, filmed from above, the head is localized and its orientation in the horizontal plane is estimated. The common marmoset has a relatively small and light head and thus low inertial weight, this means that they rely more on head movements to direct their gaze than to larger primates, such as macaques and humans. Thus, knowing the head orientation tells us more the gaze direction than it would in humans who direct their gaze much more with saccades (eye movements independent of the head). That is, these routines to estimate where the subject is and roughly where it is looking using only a webcam.

## How it works
Video frames are scaled down from 480 x 640 pixels to 76 x 102 pixels. 

The head is localized in two steps. First, the head is localized to a 52 x 52 pixel sub-region of through dividing up the 76 x 102 frame into 12, 50% overlapping, 48x48 pixel windows. The weighted average position (weighted by softmax output) is used as a center of the 52 x 52 sub-region. In the second step, a 32 x 32 pixed window is slid over the sub-region in steps of 2 pixels (i.e. 94% overlap) and the weighted average position is taken as the final position estimate.

Given a position estimate the head orientation is estimated as the weighted average of 10, 40 x 40 windows located with random offsets in the range -10 to 10 pixels on the position estimate. Orientations are binned into 12 directions spanning 0-360&deg; (i.e. in bins of  30&deg;).

## Example usage

####Label training data

```python
from data_preparation import LabelFrames
video_fname = '/full/path/to/some/video/file.mkv'
lf = LabelFrames(video_fname)

lf.run_batch(t0=2.3)   # time of 1st frame to label in seconds
```

####Extract training data and store in tfrecords format

```python
log_dir = '/directory/where/log/files/are/stored/'
video_dir = '/directory/where/video/files/are/stored/'
out_base_fname = 'name'  # TODO names will be like this...
out_dir = '/directory/where/training/and/dev/data/will/be/stored'

labeledData2storage_mrp(log_dir, video_dir, out_base_fname, out_dir)
```

#### Train the position model
```python
data_dir = '/path/to/directory/containing/tfrecords'
model_dir = '/path/to/directory/where/models/will/be/saved'
from head_tracker import TrainPositionModel
tpm = TrainPositionModel(data_dir=data_dir,
						 model_dir=model_dir)
tpm.train(level=1,Nepoch=100)
```
Wait a couple of hours.

#### Train the orientation model
```python
from head_tracker import TrainOrientationModel
tom = TrainOrientationModel(Nclass=12)
tom.train(Nepoch=50)
```
Wait a bit more.

#### Test tracking performance
```python
from head_tracker import HeadTracker
log_dir = '/directory/where/log/files/are/stored/'
video_dir = '/directory/where/video/files/are/stored/'
ht = HeadTracker()
est_track,  true_track,  error = ht.test_track_head(log_fname, video_dir, Nframe=1000)
ht.track2video(in_fname, out_fname, est_track)
```
Watch the resulting video.

## Performance
TODO


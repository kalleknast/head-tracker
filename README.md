# head-tracker
Head and gaze tracking of unrestrained marmosets using a convolutional neural network.
This repository contains the implementation of the method described in [Head and gaze tracking of unrestrained marmosets](http://biorxiv.org/content/early/2016/12/29/079566)
by Turesson, Ribeiro Conceicao and Ribeiro (2016).

## What it is
Current sophisticated technologies for manipulating and recording the nervous system allow us to perform historically unprecedented experiments. However, any influences of our experimental manipulations might have on psychological processes must be inferred from their effects on behavior. Today, quantifying behavior has become the bottleneck for large-scale, high throughput, experiments.

With **head-tracker** we try to solve this issue by using a modern deep learning algorithm for video-based animal tracking. **head-tracker** provides methods for tracking head position and direction (i.e. indirectly gaze) from simple video recordings of the common marmoset (*Callithrix jacchus*).

## What it does

The routines in this repository allows researchers to estimate where a subject is and roughly where it is looking using only a webcam. Given a video of a marmoset, filmed from above, the head is localized and its direction in the horizontal plane is estimated. The common marmoset has a relatively small and light head and thus low inertial weight, which means that they rely more on head movements to direct their gaze than to larger primates, such as macaques and humans. Thus, the head direction give a much better estimate of the gaze than it would, for example, in humans who direct their gaze much more with head-independent eye movements.

## How it works
To generate the training data, video frames are manually labeled with head direction.

Video frames are resized (480 x 640 to 160 x 120 pixels) and whitened.

The network is a CNN with six convolutional layers and two max pool layers. Filter sizes in the convolutional layers decrease from 11 × 11 to 3 × 3. The final layer is a global average pooling layer from where head position is read out from a class activity map (see: [Learning Deep Features for Discriminative Localization](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Zhou_Learning_Deep_Features_CVPR_2016_paper.html)). This global average pooling layer has a size of 1024 and is connected to a softmax output layer. We optimized the cross-entropy loss regularized by the Euclidean (L2) norm of the weights set to 0.0001.

## Example usage

#### Label training data

```python
import data_preparation as dp
video_fname = '/path/to/some/video/file.mkv' # doesn't have to be .mkv
lf = dp.LabelFrames(video_fname)

lf.run_batch(t0=2.3)   # time of 1st frame to label in seconds
```

#### Extract training data and store in tfrecords format

```python
log_dir = '/path/to/log/files/'
video_dir = '/path/to/video/files/'
out_dir = '/path/to/directory/where/training/and/dev/data/will/be/stored'

dp.labeledDataToStorage(log_dir, video_dir, data_dir, Ndev=3000)
```

#### Train the model
```python
from head_tracker import TrainModel
data_dir = '/path/to/directory/containing/tfrecords'
model_dir = '/path/to/directory/where/models/will/be/saved'
Nepoch = 80
Nclass = 29
tm = TrainModel(Nclass=Nclass,  data_dir=data_dir,  model_dir=model_dir)
valid_acc, train_acc = tm.train(Nepoch) # Train for Nepoch iterations over the traing data.
```
Be patient.


#### Test tracking performance
```python
from head_tracker import HeadTracker
log_fname = '/path/to/log/file.txt'
video_dir = '/path/to/video/files/'
video_fname = '/path/to/video/file.mkv' # doesn't have to be .mkv
video_save_fname = '/path/to/annotated/video.mp4'
ht = HeadTracker(Nclass=Nclass, model_dir=model_dir)
est_track, true_track, error = ht.test_track(log_fname, video_dir)
# write a video annotated with head direction and position.
ht.track2video(video_fname, video_save_fname, est_track)
ht.close()
```
Watch the resulting video.

## Performance
A subset of the video frames were annotated by two investigators. We use the disagreement between the two invesigators as a reference for the model's performance.
The mean head direction error was 10.9&deg; (median error 6.2&deg;), which is very close to the inter-human disagreement with a mean of 9.2&deg; (median 7.2&deg;). The mean position error was 33 pixels (median 23 pixels), compared to the inter-human
disagreement with a mean of 10 pixels (median 9 pixels).

<img src="https://github.com/kalleknast/head-tracker/blob/master/tracking_performance.png" width="630" />

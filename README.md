# head-tracker

## What it is
A common problem in research with animals is to track their behavior. The standard approaches are either to use a simple automatic, but unspecific measure, (example ...) or a specific, but labour intensive, behavioral test (example...). **head-tracker** is a few routines that allow tracking position and approximate gaze direction from simple video recordings of the common marmoset (*Callithrix jacchus*). This could allow for both very specific behavioral measures and automatic data acqusition.

## What it does

Given a video of a marmoset filmed from above it, localizes the marmoset's head, and estimates the head orientation in the horizontal plane. The common marmoset has a relatively small and light head and thus low inertial weight, this means that they rely more on head movements to direct their gaze than to larger primates, such as macaques and humans. Thus, knowing the head orientation tells us more the gaze direction than it would in humans who direct their gaze much more with saccades (eye movements independet of the head). That is, these routines to estimate where the subject is and roughly where it is looking using only a webcam.

## How it works
Multi resolutional pyramid and convnets.

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

#### Train the orientation model

#### Test tracking performance

## Results


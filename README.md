# Anthropometrics
An opensource device &amp; software for measuring different physical attributes using camera &amp; deep learning tools.

# Setup
## Dev-board
Follow instructions from the [coral website](https://coral.ai/docs/dev-board/get-started) to set up the dev board. This project requires the gstreamer python bindings to be installed, which can be done as follows:
```
sudo apt install python3-gst-1.0
```
Clone this repository, which includes the [posenet](https://github.com/google-coral/project-posenet) submodule.
## Calibration
Use the provided camera_calibration.py script to calibrate the two cameras that will be used. After calibration, the cameras should remain in the same orientation and position relative to eachother, or else will need to be recalibrated.

Print out a chessboard image such as [this one](https://github.com/opencv/opencv/blob/master/doc/pattern.png).

Note: if using a chessboard image with different dimensions, you may have to change the SIZE tuple in the python script to match the dimensions of the chessboard.

Images of the chessboard should be taken from many different angles, distances, and orientations with both cameras simultaneously. Around 15 images should work, but more images will result in a more accurate calibration.

Images from the left and right camera should be saved in different directories, but each corresponding left and right image should have the same name in order for the calibration script to work correctly.


# Usage

## camera_calibration&#46;py
```
python3 camera_calibration.py LEFT_DIRECTORY RIGHT_DIRECTORY OUTPUT_NAME SQUARE_SIZE
```
Where `LEFT_DIRECTORY` is the directory containing calibration images from the left camera, `RIGHT_DIRECTORY` is the directory containing calibration images from the right camera, `OUTPUT_NAME` is the name of the .npz file to which the calibration data should be saved, and `SQUARE_SIZE` is the size of each square on the chessboard used in the calibration photos.
See setup instructions above for details on calibration photos.

## GstPosenet&#46;py

This plugin only works if there is only one person detected in the frame. If there are more people, it may not work as expected.
```
gst-launch-1.0 STEREOSRC ! GstPosenet calibration='CALIBRATION.npz' ! SINK
```

This is a gstreamer plugin that takes a stereo video `STEREOSRC` as input and outputs the same video, printing out the height calculated at each frame. The property calibration should be set as the path to the calibration data saved by running camera_calibration.py.

If you do not have a stereo video (left and right video streams next to eachother in one video), you can use the following gstreamer pipeline to create one:

```
RIGHTSRC ! videobox border-alpha=0 left=-720 ! videomixer name=mix ! GstPosenet calibration='CALIBRATION.npz' LEFTSRC ! videobox ! mix.
```

Where `LEFTSRC` and `RIGHTSRC` are the left and right camera videos.

The height is averaged using an exponential moving average, and outliers are clipped to be within one standard deviation of the mean. This means that if someone leaves the frame and a new person walks in, it will take a few seconds for the height to stabilize around this new height again.



As we are using posenet, the best estimate of the height that we can achieve is measuring from the ankle to the eye, so measurement will always be slightly less than the true value. If distance between other keypoints (shoulders, hands, etc.) is desired, the code can be changed quite simply to use these keypoints instead.


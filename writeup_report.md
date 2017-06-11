**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[hist_steering]: ./images/distribution_hist.png "Steering Histogram"
[model_summary]: ./images/model_summary.png "Model Summary"
[center1]: ./images/center_2017_06_04_13_31_49_902.jpg "Center Lane 1"
[center1]: ./images/center_2017_06_04_13_31_49_902.jpg "Center Lane 1"
[recovery]: ./images/center_2017_06_04_15_47_27_269.jpg "Recovery Pass"
[tracktwo]: ./images/center_2017_06_04_15_55_33_842.jpg "Track Two"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Included files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md this writeup

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car drive autonomously around the track

#### 3. Submission code is usable and readable

The model.py file has the code and parameters used to train my network and save it to file.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My model is the convolution neural network designed by Nvidia referenced in the class.  I found further information for my implementation here:
https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
https://github.com/0bserver07/Nvidia-Autopilot-Keras/blob/master/model.py

In my early tests with my recorded data the model liked to go straight or, strangely enough, veer hard right.  In going through the discussion boards and conversing with my program mentor it seemed to be primarily because of an over-abundance of straight-line data.  I generated a histogram of my steering angle:

![alt text][hist_steering]

Based on this that confirmed it for me.  I ended up dropping 70% of the data under a steering angle of 0.1.  There is probably a better way to do this but this worked for me.

I also used early stopping with my Keras generator to stop training when the validation loss wasn't changing very much.

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 2. Final Model Architecture

The model begins with normalization, several convolutions then has a flatten layer and finally several fully connected layers.  Based on Nvidia's blog post, the convolutions perform the feature extraction and the fully connected layers provide the steering (although you cannot exactly state which portions of the network are doing what because it is an end-to-end network).

![alt text][model_summary]

#### 3. Creation of the Training Set & Training Process

I began by recoring multiple passes on the first track in both directions of center lane driving.
![alt text][center1]

After training and validating a network on this, I recorded some "recovery passes" from both directions of the vehicle returning to the road from off the side.
![alt text][recovery]

I also recorded a pass in track two to attempt to get the data to generalize better.
![alt text][tracktwo]


And additional recordings only in the curves

Data was augmented by flipping each image and inverting the angle to add additional training points.

In my generator data was shuffled using sklearn's shuffle utility.  20% of the information was placed into the validation set.

### Conclusions
In driving the model tends to hug the line and jerk back and forth a little bit within the lane.  I wonder if combining the CV lane finding techniques from project 1 with the neural network could help it to identify the edges better or if it's better to just let the model learn that.
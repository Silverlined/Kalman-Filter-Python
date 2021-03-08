# Kalman Filter - 1D motion example - Assignment 1

## 1. What is a Kalman Filter?
The kalman filter is a probabilistic state estimator technique and it is used to make an optimal estimation of the state of a dynamical system. It can also be explained as a recursive algorithm which consists of two main steps - Prediction & Correction. The prediction step takes, for example, the steering information of a vehicle or the motion control commands into account, in order to estimate, in order to predict where the system will be at the next point time. The correction step takes into account the sensor observation, in order to correct for potential errors in our  predictions.
The Kalman filter makes two important assumptions. The first one is that everything is Gaussian, i.e. sensor observations, errors, and noise follow a Gaussian (normal) distribution. The second assumption is that all models are linear, i.e. the model that estimates where the system will be at the next point in time and the observation model of the sensor data are both linear models. 
However, it is important to realize that we live in a non-linear world where non-Gaussian distributions are the standard. The Extended Kalman Filter (EKF) is a variant of the Kalman Filter and tries to deal with these nonlinearities. What it does is basically performing local linearization via Taylor expansion. So it performs a Taylor approximation of the non-linear models and turns them into linear ones given a current linearization point. 

## 2. Flowchart of the Kalman Filter
<img src="res/blockdiagram.png" width="640">



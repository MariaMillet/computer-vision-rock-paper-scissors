# Computer Vision RPS
Computer vision model. 

[Teachable machine](https://teachablemachine.withgoogle.com/) was used to train a computer vision model that detects whether the user is showing Rock, Paper or Scissors to the camera. Model training was done by recording frames with respective signs with locating and angles varied to capture as many variations of the same sign as possible. The model and labels were downloaded in Tensorflow model.

The virtual environment was set up using ```python'''conda create -n rps_enc''' and packages installed included: tensorflow, opencv_python, ipykernel

# manual_rps.py
Creates a Python script that simulates a Rock-Paper-Scissors game, in which the user manually inputs his choice of a sign while a computer chooses a sign randomly. 
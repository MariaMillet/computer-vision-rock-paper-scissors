# Computer Vision RPS
Computer vision model. 

[Teachable machine](https://teachablemachine.withgoogle.com/) was used to train a computer vision model that detects whether the user is showing Rock, Paper or Scissors to the camera. Model training was done by recording frames with respective signs with locating and angles varied to capture as many variations of the same sign as possible. The model and labels were downloaded in Tensorflow model.

The virtual environment was set up using ```python'''conda create -n rps_enc''' and packages installed included: tensorflow, opencv_python, ipykernel

# manual_rps.py
Creates a Python script that simulates a Rock-Paper-Scissors game, in which the user manually inputs his choice of a sign while a computer chooses a sign randomly. 

# cameras_rps.py

Create a Rock_Paper_Scissors class which uses a computer vision model trained on [Teachable machine](https://teachablemachine.withgoogle.com/) to recognise what sign the user choose to play. 

```python get_prediction()``` is the key function of the class - it processes the frame (image) with the user's choice by deploying a pre-trained model to classify a user interntion into 4 classes: Rock, Scissors, Paper and Neutral. Each time a user and a computer play the game a user gets a 3 seconds heads-up, which is displayed as the countdown on the screen. At the end of the countdown the user is prompted to make his/her choice, which is classified by the trained model. 

```python play()``` function calls ```python get_prediction``` and ```python get_winner()``` functions untill one of the players reaches a cumulative win score of 3. ```python get_winner()``` is a function which prescribes the rules of the game and allocates a winning score either to a user or a computer in case there is no tie. In case the user chosen sign is "Neutral" - i.e. no choice was made a computer wins.
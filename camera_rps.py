from itertools import count
import cv2
from keras.models import load_model
import numpy as np
import time
import random

class Rock_Paper_Scissors:
    '''
        Rock-Paper-Scissors is a two-players game, where both players choose one of three signs
        and the winner is determined according to a predefined set of rules.
        It starts with a default number of victories needed to win the game.

        
        Parameters:
        ----------
        victories: integer
            Cumulative number of wins to claim a victory in the game

        
        Attributes:
        ----------
        model: keras model
            Saved computer vision model, trained on images of Rock/Paper/Scisors
        data: numpy array
            A single frame with parameters (224, 224, 3)
        prediction_to_sign: dictionary
            Mapping between the label integer and the label
        predicted_sign: string
            The user's sign predicted by the model
        computer_wins: intger
            Total number of wins for a computer
        user_wins: integer
            Total number of wins for a human-player
        victories: integer
            Cumulative number of single wins needed to win the game

        Methods:
        -------
        draw_text(frame, text, x, y, color, thickness, size)
            Puts text on the frame in a pre-specified space with prespecified size/thickness/color.
            Used to output the countdown for the user and notify him/her of the score after each game.
        get_prediction()
            Processes the frame where the user shows a chosen sign and updates a predicted_sign attribute.
        get_winner (computer_choice, user_choice)
            Determines whether a computer or a user wins, or neither. Updated computer_wins, user_wins attributes
        play()
            Sets the play in motion until the number of victories of one of the players reaches victories attribute.
        '''
    
    def __init__(self, victories=3):        
        self.model = load_model('keras_model.h5')
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.prediction_to_sign = {}
        with open('labels.txt') as f:
            for line in f:
                (key, val) = line.split()
                self.prediction_to_sign[int(key)] = val
        self.predicted_sign = None
        self.computer_wins = 0
        self.user_wins = 0
        self.victories = victories

    def draw_text(self,frame, text, x, y, color=(255,0,255), thickness=10, size=3):
        ''' Draws a text string on an image (frame) '''
        if x is not None and y is not None:
            cv2.putText(
            frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

    def get_prediction(self):
        ''' Predicts a sign of the user from a frame '''
        
        cap = cv2.VideoCapture(0)
        # count_down sets the end time for the camera to record a user input. 
        # Here we allows for a total of 5 seconds of "filming", 3 of which are used for a countdown.
        end_of_filming = time.time() + 5
        start_time = time.time()
        # counter variable determines a countdown time in seconds
        counter = 3
        while end_of_filming > time.time():
            ret, frame = cap.read()
            if ret == True:
                center_x = int(frame.shape[0]/2)
                center_y = int(frame.shape[0]/2)
                # a countdown message is displayed
                if time.time() - start_time >= 1 and counter > 0:
                    self.draw_text(frame, "Live in " + str(counter), center_x, center_y)
                    counter -= 1
                    start_time = time.time()
                elif counter > 0:
                    self.draw_text(frame, "Live in " + str(counter), center_x, center_y)
                # after the countdown "Play" message is displayed
                else:
                    self.draw_text(frame, "Play!", center_x, center_y)

                resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
                image_np = np.array(resized_frame)
                # Normalize the image
                normalized_image = (image_np.astype(np.float32) / 127.0) - 1 
                self.data[0] = normalized_image
                prediction = self.model.predict(self.data)
                cv2.imshow('frame', frame)
                # cv2.displayOverlay('frame', text="hello")
                # Press q to close the window
                print(prediction)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                prediction_index = np.argmax(prediction[0])
                self.predicted_sign = self.prediction_to_sign[prediction_index]
                print(f"you chose {self.predicted_sign}")
        # After the loop release the cap object
        cap.release()
        # Destroy all the windows
        cv2.destroyAllWindows()


    def get_winner(self, computer_choice, user_choice):
        ''' Predicts a sign of the user from a frame 
        
        Args:
            computer_choice (str): A sign randomly chosen by the computer
            user_choice (str): A sign shown by the user
        '''
        if (computer_choice == "Paper" and user_choice == "Rock") or \
            (computer_choice == "Scissors" and user_choice == "Paper") or \
            (computer_choice == "Rock" and user_choice == "Scissors") or \
            user_choice == "Nothing":
            print("Computer wins")
            # update the computer score
            self.computer_wins += 1
        elif (computer_choice == "Rock" and user_choice == "Paper") or \
            (computer_choice == "Paper" and user_choice == "Scissors") or \
            (computer_choice == "Scissors" and user_choice == "Rock"):
            print("User wins")
            # update the user score
            self.user_wins += 1
        else:
            print("Tie")

    def play(self):
        while self.computer_wins != self.victories and self.user_wins != self.victories:
            self.get_prediction()
            # define a list of possible signs for computer to choose from. Note excludes "Nothing" class
            rps_list = list(self.prediction_to_sign.values())[:3]
            computer_choice = random.choice(rps_list)
            print(f"Computer choice is {computer_choice}")
            self.get_winner(computer_choice, self.predicted_sign)
            # pausing the next game by 5 seconds and showing the score
            count_down = time.time() + 5
            cap = cv2.VideoCapture(0)
            while count_down > time.time():
                ret, frame = cap.read()
                center_x = int(frame.shape[0]/10)
                center_y = int(frame.shape[0]/10)
                self.draw_text(frame, f"Computer score: {self.computer_wins} Your score: {self.user_wins} \
                                    Next game in 5s", center_x, center_y)
                center_y = int(frame.shape[0]/4)
                self.draw_text(frame, f"Next game in 5s", center_x, center_y)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # After the loop release the cap object
            cap.release()
            # Destroy all the windows
            cv2.destroyAllWindows()
    
        if self.computer_wins == 3:
            print("Computer won!")
        else:
            print("You won!")


game = Rock_Paper_Scissors()
game.play()



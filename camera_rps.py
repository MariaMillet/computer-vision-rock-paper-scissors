from itertools import count
import cv2
from keras.models import load_model
import numpy as np
import time
import random
model = load_model('keras_model.h5')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


def draw_text(frame, text, x, y, color=(255,0,255), thickness=10, size=3):
            if x is not None and y is not None:
                cv2.putText(
                    frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

prediction_to_sign = {0: "Rock", 1: "Paper", 2: "Scissors", 3: "Nothing"}

def get_prediction():
    counter = 3
    cap = cv2.VideoCapture(0)
    prediction_to_sign = {0: "Rock", 1: "Paper", 2: "Scissors", 3: "Nothing"}
    # count_down sets the end time for the camera to record a user input
    count_down = time.time() + 5
    start_time = time.time()
    while count_down > time.time():
        ret, frame = cap.read()
        if ret == True:
            center_x = int(frame.shape[0]/2)
            center_y = int(frame.shape[0]/2)
            if time.time() - start_time >= 1 and counter > 0:
                draw_text(frame, "Live in " + str(counter), center_x, center_y)
                counter -= 1
                start_time = time.time()
            elif counter > 0:
                draw_text(frame, "Live in " + str(counter), center_x, center_y)
            else:
                draw_text(frame, "Play!", center_x, center_y)

            resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
            image_np = np.array(resized_frame)
            normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
            data[0] = normalized_image
            prediction = model.predict(data)
            cv2.imshow('frame', frame)
            # cv2.displayOverlay('frame', text="hello")
            # Press q to close the window
            print(prediction)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            prediction_index = np.argmax(prediction[0])
            predicted_sign = prediction_to_sign[prediction_index]
            print(f"you chose {predicted_sign}")
    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    return predicted_sign

def get_winner(computer_choice, user_choice):
    if (computer_choice == "Paper" and user_choice == "Rock") or \
        (computer_choice == "Scissors" and user_choice == "Paper") or \
        (computer_choice == "Rock" and user_choice == "Scissors") or \
        user_choice == "Nothing":
        print("Computer wins")
        return (1,0)
    elif (computer_choice == "Rock" and user_choice == "Paper") or \
        (computer_choice == "Paper" and user_choice == "Scissors") or \
        (computer_choice == "Scissors" and user_choice == "Rock"):
        print("User wins")
        return (0,1)
    else:
        print("Tie")
        return (0,0)

def play(victories = 2):
    computer_wins = 0
    user_wins = 0
    while computer_wins != victories and user_wins != victories:
        user_choice = get_prediction()
        # define a list of possible signs for computer to choose from. Note excludes "Nothing" class
        rps_list = list(prediction_to_sign.values())[:3]
        computer_choice = random.choice(rps_list)
        print(f"Computer choice is {computer_choice}")
        computer_score, user_score = get_winner(computer_choice, user_choice)
        computer_wins += computer_score
        user_wins += user_score     
        count_down = time.time() + 5
        cap = cv2.VideoCapture(0)
        while count_down > time.time():
            ret, frame = cap.read()
            center_x = int(frame.shape[0]/10)
            center_y = int(frame.shape[0]/10)
            draw_text(frame, f"Computer score: {computer_wins} Your score: {user_wins} \
                                Next game in 5s", center_x, center_y)
            center_y = int(frame.shape[0]/6)
            draw_text(frame, f"Next game in 5s", center_x, center_y)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # After the loop release the cap object
        cap.release()
        # Destroy all the windows
        cv2.destroyAllWindows()
   

    
    if computer_wins == 3:
        print("Computer won!")
    else:
        print("You won!")

play()



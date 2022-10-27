# %%
import random

def get_computer_choice():
    return random.choice(["Rock", "Paper", "Scissors"])

def get_user_choice():
    return input("Please input your choice.")

def get_winner(computer_choice, user_choice):
    if (computer_choice == "Paper" and user_choice == "Rock") or \
        (computer_choice == "Scissors" and user_choice == "Paper") or \
        (computer_choice == "Rock" and user_choice == "Scissors"):
        print("Computer wins")
    else:
        print("User wins")

# %%
get_winner("Paper", 'Scissors')
# %%

def play():
    get_winner(get_computer_choice(), get_user_choice())
# %%
play()
# %%

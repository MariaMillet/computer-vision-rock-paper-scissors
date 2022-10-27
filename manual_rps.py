import random

def get_computer_choice():
    return random.choice(["Rock", "Paper", "Scissors"])

def get_user_choice():
    return input("Please input your choice.")

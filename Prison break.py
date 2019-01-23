#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:37:48 2018

@author: Frederic Van Kelecom


DocString:
    
    A) Introduction:
    You're a death row inmate of the the San Quintin State Prison. As tomorrow 
    is the day of your execution, today is the last day that you can save your 
    life by escaping the prison. In order to escape, you will have to complete 
    a series of decisions and games. The only way to escape is by successfully 
    completing all stages. The game has three stages:
    1. Decision: Attack the guard or not
    2. Game: Rock, paper, sciccors against a rival gang member (2 chances)
    3. Lock: Guess the right code of a 4 digit lock (3 chances)

    Every wrong decision or every loss in a game will lead to getting caught
    and ultimately death.

    B) Known Bugs and/or Errors:
       None.

"""
from sys import exit
from random import randint

global name

def start():
    print("\f")

    print("Prison Break")

    print("\nWhat is your name?", end = ' ')
    name = input("> ")

    print(f"""
    Hello {name}! You are an inmate of the San Quintin State Prison, one of the 
    most notorious prisons in the United States. You have been locked up for 10 
    years for a first degree murder on your gardener for sleeping with your 
    wife. Currently, you are serving time on death row and tomorrow is the day 
    of your execution. In order to avoid being put to death, you will have to 
    escape today! The only way to succesfully escape is by making consequently 
    the right choices.
    """)

    input("\n <Press enter to start your escape >\n")
    decision()

#decision: attack or not
def decision():

    print(f"""
    It is 2pm in the afternoon and you are lying on your bed. You hear the
    footsteps of a guard approaching your cell. What do you do?
      1. Attack the guard and steal his keys.
      2. Patiently wait for the guard and see what he does.
      """)
     
    dec1 = input("> ")
    
    if dec1 == "attack" or dec1 == "1":
        print("The guard overpowered you and you lost your privilege for a break today.")
        input("\n <Press enter to continue >\n")
        print("\f")
        fail()
        
    elif dec1 == "wait" or dec1 == "2":
        print("""The guard says: "It is time for your break! Get out of your cell!"
              and he escorts you out of your cell to the courtyard.""")
        input("\n <Press enter to continue >\n")
        print("\f")
        game()
        
    else:
        print("Invalid command. Please try again.")
        input("<Press enter to continue>\n")
        decision()
        
#game: rock, paper scissors
def game():
    print(f"""
You walk to the courtyard and you bump into a member of notorious RPS gang. 
In order to bypass him, you'll have to beat him in a game of rock, paper, scissors!
You have 2 chances in total! What's your guess?
             1) rock
             2) paper
             3) scissors
    
""")
    chances = 2
    while chances > 0:
        flip_value = randint(1,3) # 1 = rock, 2 = paper, 3 = scissors 
        choice = input("> ")
        if "1" in choice or "rock" in choice:
            if flip_value == 1:
                print(f"""
You both chose rock! Try again!
                      
                      """)
            elif flip_value == 3:
                print(f"""
Your opponent chose scissors! You won!!!

                """)
                input('<Press any key to continue>\n')
                lock()
                break
            
            else:
                print(f"""
You lose! Your opponent chose paper!

""")
                chances -= 1

        elif "2" in choice or "paper" in choice:
            if flip_value == 2:
                print(f"""
You both chose paper! Try again!
                      
                      """)
            
            elif flip_value == 1:
                print(f"""
Your opponent chose rock! You won!!!
                      
                      """)
                input('<Press any key to continue>\n')
                lock()
                break
            
            else:
                print(f"""
You lose! Your opponent chose scissors!

""")
                chances -= 1
                
        elif "3" in choice or "scissors" in choice:
            if flip_value == 3:
                print(f"""
You both chose scissors! Try again!
                      
                      """)
            elif flip_value == 2:
                print(f"""
Your opponent chose paper! You won!!!

""")
                      
                input('<Press any key to continue>\n')
                lock()
                break
            
            else:
                print(f"""
You lose! Your opponent chose rock!

""")
                chances -= 1

        else:
            print("Invalid command. Please try again.")
            input("<Press enter to continue>\n")
            
    fail()
 
#pick the lock
def lock():
    print(f"""
You passed the RPS gang member and you've come to the final stage:
Guess the code of the lock of the door that leads to the outside world!
Be aware, you have 3 attempts before the guard sees and catches you!
HINT: The code is the year that San Quintin State Prison opened!
          """)
    attempts = 3
    while attempts > 0 :
        number = 1852    
        print("Guess the 4-digit code! Amount of chances: " + str(attempts))
        guess = inputNumber("> ")
        guess = str(guess)
        if len(guess) == 4:
            if int(guess) == number:
                print(f"""you guessed the right code! Congratulations, you are free!!
              """)
                exit(0)
                break
            else:
                numbero = str(number)
                for element in range(0,4):
                    if guess[element] == numbero[element]:
                        print("digit number " + str(element+1) + " is correct")
            
                    elif int(guess[element]) > int(numbero[element]):
                        print("digit number " + str(element+1) + " is lower")
            
                    else:
                        print("digit number " + str(element+1) + " is higher")
                attempts -= 1
        else:
            print("Invalid command. Please try again.")   
    fail() 
    
#fail function
def fail():
    print(f"""
You were caught and therefore you failed to escape the prison! Tomorrow you
will die knowing that you've tried everything!"            
    """)
    
    print(f"Would you like to play again? (Yes/No)")
    replay = input("> ")
    replay = replay.lower()
    
    if replay == 'yes':
        start()
        
    else:
        print("f: Thanks for playing!")
        exit(0)
        
#integer enforcement
def inputNumber(message):
  while True:
    try:
       userInput = int(input(message))       
    except ValueError:
       print("Not an integer! Try again.")
       continue
    else:
       return userInput 
       break
        
###############################################################################
# Game Start
###############################################################################
start()
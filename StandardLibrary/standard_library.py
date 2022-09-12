import calculator
import box
from itertools import chain, combinations
import random
import time
import sys

# standard_library.py
"""Python Essentials: The Standard Library.
<Name> Trevor Wai
<Class> Section 2
<Date> 9/6/22
"""


# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order, separated by a comma).
    """
    #returns min max and average of elements from list L
    return min(L), max(L), sum(L)/len(L)


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test integers, strings, lists, tuples, and sets. Print your results.
    """
    #Assings a to an int
    a = 1
    #Creates a copy of a
    b = a
    #Updates a
    a += 1
    #Tests to see if a and b are equivilent
    print(a == b)
    
    #Assins string to a
    a = 'Hello World!'
    #creates a copy of a
    b = a
    #updates a
    a += ' Goodbye World!'
    #tests to see if a and b are equivilent
    print(a == b)

    #Creates list a
    a = [1,2,3]
    #copies list a
    b = a
    #updates a
    a.append(4)
    #tests to see if a and b are still equal
    print(a == b)


    #Creates tuple a
    a = (4,5,6)
    #copies a
    b = a
    #updates a
    a += (1, )
    #checks to see if a and b are equal
    print(a == b)
    
    #creates set a
    a = {1,2,3}
    #copies a
    b = a
    #updates a
    a.add(4)
    #checks to see if a and b are equal
    print(a == b)

    #Prints the answer to which are mutable and immutable
    print("Mutable: lists, sets Immutable: int, string, tuple")


# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt() that are
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    #calculates the hypotenuse using the calculator module
    hyp = calculator.sqrt(calculator.sum(calculator.prod(a,a), calculator.prod(b,b)))
    #returns the answer
    return hyp


# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    #Creates an empyt list
    L = []

    #for loop to assigns subsets of A to the list
    for i in range(len(A) + 1):
        #Updates the list L with the set of subsets of A
        L += [set(j) for j in list(combinations(A,i))]

    #returns the list
    return L


# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""
    #Creates the list of remaining numbers
    remaining = [1,2,3,4,5,6,7,8,9]
    #creates a int variable from the string timelimit the user inputs
    time_left = int(timelimit)
    #Initializes the win condition to be false
    win = False
    #Starts the clock
    start = time.time()
    #initializes the score to be equal to the number of remaining numbers
    score = 45

    #while loop that runs as long as there is still time left
    while time_left > 0:

        #If the score is greater than 6 then rolls two dice
        if score > 6:
            roll = random.randint(2,12)
        #if the score is less than 6 then rolls one die
        else:
            roll = random.randint(1,6)

        #Prints the list of remaining numbers
        print('Numbers left: ', remaining)
        #Prints the roll
        print('Roll: ', roll)
        #prints the time left
        print('Seconds left: ', time_left)
        #Checks to see if a roll is value
        if box.isvalid(roll, remaining) == True:
            #accepts a user input of numbers to eliminate
            user_input = input('Numbers to eliminate: ' )
            #assigns the list returned from parse_input to the variable x
            x = box.parse_input(user_input, remaining)
            
            #While the list is empty runs a loop saying inputs are invalid
            while len(x) == 0:
                #Continues counting down the time
                time_left = round(int(timelimit) - (time.time()- start), 2)
                #prints the time left
                print('Seconds left: ', time_left)
                #Informs player that the input is invalid
                print('Invalid input')
                #has the user input new numbers and assigns the list returned from parse input and assigns it to variable x
                x = box.parse_input(input('Numbers to eliminate: ' ), remaining)

            #while the numbers inputed don't add up to the roll runs a loop saying the input is invalid
            while sum(x) != roll:
                #Continues counting down time
                time_left = round(int(timelimit) - (time.time()- start), 2)
                #prints the time
                print('Seconds left: ', time_left)
                #informs player that the input is invalid
                print('Invalid input')
                #Has the user input new nmbers and assigns the lsit returned from parse input and assings it to a variable
                x = box.parse_input(input('Numbers to eliminate: ' ), remaining)
            
            #A for loop removing the numbers the player inputs
            for i in range(len(x)):
                #removes the numbers the player inputs
                remaining.remove(x[i])
                #updates the score
                score -= sum(x)

        #Breaks if there are no remaining numbers                    
        elif remaining == []:
            #changes the win condition to true
            win = True
            #leaves the while loop
            break
        
        #If there is no way to add numbers to the sum
        else:
            #Assigns the win condition to false
            win = False
            #leaves the while loop
            break
        
            

        #finds the end time
        end = time.time()
        #updates the time left
        time_left = round(int(timelimit) - (end - start), 2)
    
    #Computes the amount of time it took for the player to play the game
    time_played = int(timelimit) - time_left


    #If the win condition was changed to true then outputs a value
    if(win == True):
        #prints a new line
        print('\n')
        #Prints the player name, score, time played,and a congratulations message
        print("Score for player", player, ": ", score, " points", "\n", "Time Played:", round(time_played, 2), " seconds", '\n', 'Congratulations!! You shut the box!')

    #If the win condition was not changed to true
    else:
        #calculates the score
        for i in range(len(remaining)):
            score += remaining[i]
        #Prints the player name, score and time played a a failure message
        print("Score for player", player, ": ", score, " points", "\n", "Time Played:", round(time_played, 2), " seconds", '\n', 'Better luck next time >:)')

#Runs the shut the box game if there are three arguments
if __name__ == "__main__":
    #checks to make sure there are three arguments
    if len(sys.argv) == 3:
        shut_the_box(sys.argv[1], sys.argv[2])

#testing


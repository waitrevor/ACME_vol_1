# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
<Name> Trevor Wai
<Class> Section 2
<Date> 9/20/22
"""

from multiprocessing.sharedctypes import Value
from random import choice
import numpy as np


# Problem 1
def arithmagic():
    """
    Takes in user input to perform a magic trick and prints the result.
    Verifies the user's input at each step and raises a
    ValueError with an informative error message if any of the following occur:

    The first number step_1 is not a 3-digit number.
    The first number's first and last digits differ by less than $2$.
    The second number step_2 is not the reverse of the first number.
    The third number step_3 is not the positive difference of the first two numbers.
    The fourth number step_4 is not the reverse of the third number.
    """
    #Takes an input
    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")

    num1 = int(step_1)
    #Raises an error if the input is not 3 digits
    if num1 < 99 or num1 > 999:
        raise ValueError("The first number is not a 3-digit number")

    #Raises an error if the input's first and last digit differ by less than 2
    elif abs(int(step_1[0]) - int(step_1[2])) != 2:
        raise ValueError("The first number's first and last "
                                            "digit differ by less than 2")

    #takes the second input
    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")
    num2 = int(step_2)
    #Raises an error if the second input is not the reverse of the the first
    if step_2[0] != step_1[2]:
        raise ValueError("The second number is not the reverse of the first number")
    elif step_2[1] != step_1[1]:
        raise ValueError("The second number is not the reverse of the first number")
    elif step_2[2] != step_1[0]:
        raise ValueError("The second number is not the reverse of the first number")
    #Takes the third input
    step_3 = input("Enter the positive difference of these numbers: ")
    num3 = int(step_3)
    #Raises an error if the third input is not the positive difference of the first two numbers
    if num3 != abs(num2 - num1):
        raise ValueError("The third number is not the positive difference "
                                                "of the first two numbers")
    #takes the fourth input
    step_4 = input("Enter the reverse of the previous result: ")
    #raises an error if the fourth number is not the reverse of the third number
    if step_4[0] != step_3[2]:
        raise ValueError("The fourth number is not the reverse of the third number")
    elif step_4[1] != step_3[1]:
        raise ValueError("The fourth number is not the reverse of the third number")
    elif step_4[2] != step_3[0]:
        raise ValueError("The fourth number is not the reverse of the third number")

    #Prints the sum of the third and fourth input
    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")


# Problem 2
def random_walk(max_iters=1e12):
    """
    If the user raises a KeyboardInterrupt by pressing ctrl+c while the
    program is running, the function should catch the exception and
    print "Process interrupted at iteration $i$".
    If no KeyboardInterrupt is raised, print "Process completed".

    Return walk.
    """
    
    walk = 0
    directions = [1, -1]
    #Tries to run the random walk function
    try:
        for i in range(int(max_iters)):
            walk += choice(directions)
        print("Process completed")
        return walk
    #If there is a keyboard interupt print the process interupt at the specific iteration
    except:
        print(f"Process interupted at iteration {i}")
        return walk


# Problems 3 and 4: Write a 'ContentFilter' class.
class ContentFilter(object):
    """Class for reading in file

    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file

    """
    # Problem 3
    def __init__(self, filename):
        """ Read from the specified file. If the filename is invalid, prompt
        the user until a valid filename is given.
        """

        isOpen = False
        #While loop to test to see if a file name is valid
        while isOpen == False:
            try: 
                #if filename is valid opens the file and stores the contents as an attribute 
                with open(filename, 'r') as outFile:
                    self.contents = "".join(outFile.readlines())
                isOpen = True
            except:
                #Prompts the user to try again if the file name is invalid
                filename = input("Please enter a valid file name: ")

        #stores the file name as an attribute
        self.filename = filename
        #Stores the lines of the file as an attribute
        self.lines = self.contents.split('\n')

        




            

 # Problem 4 ---------------------------------------------------------------
    def check_mode(self, mode):
        """ Raise a ValueError if the mode is invalid. """
        #Checks to make sure that the mode input is valid
        if mode not in "wxa":
            raise ValueError("Mode is invalid")

    def uniform(self, outfile, mode='w', case='upper'):
        """ Write the data to the outfile with uniform case. Include an additional
        keyword argument case that defaults to "upper". If case="upper", write
        the data in upper case. If case="lower", write the data in lower case.
        If case is not one of these two values, raise a ValueError. """
        #Checks the mode
        self.check_mode(mode)
        #Opens a outfile to write to
        with open(outfile, mode=mode) as outfile:
            #Tests to see if the case is upper and writes to the outfile
            if case == 'upper':
                outfile.write(self.contents.upper().strip())
            #Tests to see if the case is lower and writes to the outfile
            elif case == 'lower':
                outfile.write(self.contents.lower().strip())
            #Raises an error if the case is not upper or lower
            else:
                raise ValueError("Case is not 'upper' or 'lower'")


    def reverse(self, outfile, mode='w', unit='line'):
        """ Write the data to the outfile in reverse order. Include an additional
        keyword argument unit that defaults to "line". If unit="word", reverse
        the ordering of the words in each line, but write the lines in the same
        order as the original file. If units="line", reverse the ordering of the
        lines, but do not change the ordering of the words on each individual
        line. If unit is not one of these two values, raise a ValueError. """
        #checks the mode and opens an outfile to write to
        self.check_mode(mode)
        with open(outfile, mode=mode) as outfile:
            #Checks if the unit is word
            if unit == 'word':

                #splits the string into a list of lists
                linesList = self.contents.split('\n')
                wordsList = [line.split(' ') for line in linesList]
                #Reverses the inner list
                rev_wordsList = [word[::-1] for word in wordsList]

                #Rejoins the lists to recreate a string
                rev_lineList = [' '.join(rev_word) for rev_word in rev_wordsList]
                finalList = '\n'.join(rev_lineList)

                #Writes the new string to the outfile
                outfile.write(finalList.strip())

            #Checks if the unit is line
            elif unit == 'line':

                #Splits the string into a list and rearragnes the elements
                L = self.contents.split('\n')
                L = '\n'.join(L[::-1])

                #Writes the new string to the outfile 
                outfile.write(L.strip())

            #Raises an error if the unit is not word or line
            else:
                raise ValueError("Unit is not 'word' or 'line'")

    def transpose(self, outfile, mode='w'):
        """ Write a transposed version of the data to the outfile. That is, write
        the first word of each line of the data to the first line of the new file,
        the second word of each line of the data to the second line of the new
        file, and so on. Viewed as a matrix of words, the rows of the input file
        then become the columns of the output file, and viceversa. You may assume
        that there are an equal number of words on each line of the input file. """
        #Checks the mode and creates an outfile
        self.check_mode(mode)
        with open(outfile, mode=mode) as outfile:

            #turns the string into a list of lists
            lines = self.contents.split('\n')
            words = [l.split(' ') for l in lines]

            #Transposes the list of lists
            newTrans = ''
            for i in range(len(words[0])):
                for j in range(len(lines) - 1):

                    #puts the elements into a string
                    if j == len(words) - 2:
                        newTrans += words[j][i]
                    else:
                        newTrans += words[j][i] + " "
                newTrans += "\n"

            #Writes to the outfile  
            outfile.write(newTrans.strip())
            


    def __str__(self):
        """ Printing a ContentFilter object yields the following output:

        Source file:            <filename>
        Total characters:       <The total number of characters in file>
        Alphabetic characters:  <The number of letters>
        Numerical characters:   <The number of digits>
        Whitespace characters:  <The number of spaces, tabs, and newlines>
        Number of lines:        <The number of lines>
        """
        #declares stats as a variable with all the statistics from the input file
        stats = f"Source file:\t\t{self.filename}\
        \nTotal characters:\t{len(self.contents)}\
        \nAlphabetic characters:\t{sum([letter.isalpha() for letter in self.contents])}\
        \nNumerical characters:\t{sum([num.isdigit() for num in self.contents])}\
        \nWhitespace characters:\t{sum([s.isspace() for s in self.contents])}\
        \nNumber of lines:\t{len(self.lines)}"

        #returns stats
        return stats
        


#testing
# cf = ContentFilter("hello_world.txt")
# cf.uniform("uniform.txt")
# cf.reverse("reverse.txt", mode='w', unit='word')
# cf.transpose("transpose.txt", mode='w')
# print(cf)
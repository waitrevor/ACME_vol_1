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

    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")
    num1 = int(step_1)
    if num1 < 99 or num1 > 999:
        raise ValueError("The first number is not a 3-digit number")
    elif abs(int(step_1[0]) - int(step_1[2])) != 2:
        raise ValueError("The first number's first and last "
                                            "digit differ by less than 2")

    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")
    num2 = int(step_2)
    if step_2[0] != step_1[2]:
        raise ValueError("The second number is no tthe revrse of the first number")
    elif step_2[1] != step_1[1]:
        raise ValueError("The second number is not the reverse of the first number")
    elif step_2[2] != step_1[0]:
        raise ValueError("The second number is not the reverse of the first number")

    step_3 = input("Enter the positive difference of these numbers: ")
    num3 = int(step_3)
    if num3 != abs(num2 - num1):
        raise ValueError("The third number is not the positive difference "
                                                "of the first two numbers")

    step_4 = input("Enter the reverse of the previous result: ")
    if step_4[0] != step_3[2]:
        raise ValueError("The fourth number is no the reverse of the third number")
    elif step_4[1] != step_3[1]:
        raise ValueError("The fourth number is not the reverse of the third number")
    elif step_4[2] != step_3[0]:
        raise ValueError("The fourth number is not the reverse of the third number")

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
    try:
        for i in range(int(max_iters)):
            walk += choice(directions)
        print("Process completed")
        return walk
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
        self.filename = filename

        isOpen = False

        while isOpen == False:
            try: 
                with open(self.filename, 'r') as outFile:
                    self.contents = "".join(outFile.readlines())
                isOpen = True

            except:
                self.filename = input("Please enter a valid file name: ")




            

 # Problem 4 ---------------------------------------------------------------
    def check_mode(self, mode):
        """ Raise a ValueError if the mode is invalid. """
        if mode not in "wxa":
            raise ValueError("Mode is invalid")

    def uniform(self, outfile, mode='w', case='upper'):
        """ Write the data to the outfile with uniform case. Include an additional
        keyword argument case that defaults to "upper". If case="upper", write
        the data in upper case. If case="lower", write the data in lower case.
        If case is not one of these two values, raise a ValueError. """
        self.check_mode(mode)
        with open(outfile, mode=mode) as outfile:
            if case == 'upper':
                outfile.write(self.contents.upper())
            elif case == 'lower':
                outfile.write(self.contents.lower())
            else:
                raise ValueError("Case is not 'upper' or 'lower'")


    def reverse(self, outfile, mode='w', unit='line'):
        """ Write the data to the outfile in reverse order. Include an additional
        keyword argument unit that defaults to "line". If unit="word", reverse
        the ordering of the words in each line, but write the lines in the same
        order as the original file. If units="line", reverse the ordering of the
        lines, but do not change the ordering of the words on each individual
        line. If unit is not one of these two values, raise a ValueError. """
        self.check_mode(mode)
        with open(outfile, mode=mode) as outfile:
            if unit == 'word':
                print(self.contents[::-1])
                outfile.write(self.contents[::-1])
            elif unit == 'line':
                L = ''.join(self.contents)
                print(L)
                L = self.contents.split('\n')
                print(L)
                L = L[::-1]
                print(L)
                L = ''.join(L)
                print(L)
                outfile.write(L)
            else:
                raise ValueError("Unit is not 'word' or 'line'")

    def transpose(self, outfile, mode='w'):
        """ Write a transposed version of the data to the outfile. That is, write
        the first word of each line of the data to the first line of the new file,
        the second word of each line of the data to the second line of the new
        file, and so on. Viewed as a matrix of words, the rows of the input file
        then become the columns of the output file, and viceversa. You may assume
        that there are an equal number of words on each line of the input file. """
        self.check_mode(mode)
        #with open(outfile, mode=mode) as outfile:
            


    def __str__(self):
        """ Printing a ContentFilter object yields the following output:

        Source file:            <filename>
        Total characters:       <The total number of characters in file>
        Alphabetic characters:  <The number of letters>
        Numerical characters:   <The number of digits>
        Whitespace characters:  <The number of spaces, tabs, and newlines>
        Number of lines:        <The number of lines>
        """

#testing

cf = ContentFilter("cf_example1.txt")
cf.uniform("uniform.txt", mode='w', case='upper')
cf.reverse("reverse.txt", mode='w', unit='word')

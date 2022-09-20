# object_oriented.py
"""Python Essentials: Object Oriented Programming.
<Name> Trevor Wai
<Class> Section 2
<Date> 9/13/22
"""
from cgi import test
import math

class Backpack:
    """A Backpack object class. Has a name and a list of contents.

    Attributes:
        name (str): the name of the backpack's owner.
        contents (list): the contents of the backpack.
    """

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size = 5):
        """Set the name and initialize an empty list of contents.

        Parameters:
            name (str): the name of the backpack's owner.
            color (str): the color of the backpack
            max_size (int): The max number of items the backpack can hold
        """
        self.name = name
        self.contents = []
        #Stores max_size as an attribute
        self.max_size = max_size
        #Stores color as an attribute
        self.color = color


    def put(self, item):
        """Checks to make sure the backpack isn't at max capacity.
        If the backpack isn't at max capacity Add an item to the backpack's list of contents."""
        #If statment to check to see if the backpack is at max capacity
        if len(self.contents) >= self.max_size:
            #Prints no room and doesn't add any items to the backpack
            print("No room!")
        else:
            #If the backpack is not at max capacity adds the item
            self.contents.append(item)

    def take(self, item):
        """Remove an item from the backpack's list of contents."""
        #Removes a specified item from the backpack
        self.contents.remove(item)

    def dump(self):
        """Removes all the items from the backpack's list of contents."""
        #removes all the contents of the backpack
        self.contents.clear()


    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)

    def __eq__(self, other):
        """Compares two backpack objects."""
        #returns true if the name and color and length of both backpacks are the same
        return self.name == other.name and self.color == other.color and len(self.contents) == len(other.contents)

    def __str__(self):
        """Prints the Backpack"""
        #Prints all the attributes of the backpack
        return f"Owner:\t\t{self.name}\nColor:\t\t{self.color}\nSize:\t\t{len(self.contents)}\nMax Size:\t{self.max_size}\nContents:\t{self.contents}"


# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
class Jetpack(Backpack):

    def __init__(self, name, color, fuel = 10, max_size = 2):
        #Defines constructor

        Backpack.__init__(self, name, color, max_size)
        #Initializes fuel attribute
        self.fuel = fuel

    def fly(self, amount):
        """fly method that accepts an amount of fuel and
         returns the origional fuel amount minus the input amount"""
        if self.fuel > amount:
            self.fuel -= amount
        else:
            #Returns not enough if the amount input is greater than fuel in the jetpack
            print("Not enough Fuel!")

    def dump(self):
        """Dumps all the fuel out of the backpack"""
        #Sets fuel amount to zero
        self.fuel = 0
        Backpack.dump(self)

# Problem 4: Write a 'ComplexNumber' class.

class ComplexNumber:

    def __init__(self, real, imag):
        #Defines the constructor and initializes attributes
        self.real = real
        self.imag = imag

    def conjugate(self):
        #Returns the conjugate of the complex number
        return  ComplexNumber(self.real, -self.imag)

    def __str__(self):
        #Prints complex number
        if self.imag < 0:
            return f"({self.real}{self.imag}j)"
        else:
            return f"({self.real}+{self.imag}j)"

    def __abs__(self):
        #Returns the magnitude of the complex number
        return abs(math.sqrt(self.real**2 + self.imag**2))

    def __eq__(self, other):
        #compares two complex numbers
        return self.real == other.real and self.imag == other.imag

    def __add__(self, other):
        #adds two complex numbers together
        return ComplexNumber(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        #subtracts two complex numbers
        return ComplexNumber(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        #multiplies two complex numbers
        return ComplexNumber(self.real * other.real - self.imag * other.imag, self.real * other.imag + self.imag * other.real)

    def __truediv__(self, other):
        #divides two complex numbers
        r_top = self.__mul__(other.conjugate()).real
        i_top = self.__mul__(other.conjugate()).imag
        bot = other.real**2 + other.imag**2
        #returns quotient of two complex numbers
        return ComplexNumber(r_top / bot, i_top / bot)


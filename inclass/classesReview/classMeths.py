#What is a class method?

#A class method is a method that is bound to a class rather than its object. It doesn't require creation of a class instance, much like staticmethod.

#The difference between a static method and a class method is:

#    Static method knows nothing about the class and just deals with the parameters
#    Class method works with the class since its parameter is always the class itself.

#The class method can be called both by the class and its object.

#ex 1

from datetime import date

# random Person
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    @classmethod
    def fromBirthYear(cls, name, birthYear):
        return cls(name, date.today().year - birthYear)


    def display(self):
        print(self.name + "'s age is: " + str(self.age))

person = Person('Adam', 19)
person.display()

person1 = Person.fromBirthYear('John',  1985)
person1.display()


#ex 2

#Note how I’m using the cls argument in the margherita and prosciutto factory methods instead of calling the Pizza constructor directly.

#This is a trick you can use to follow the Don’t Repeat Yourself (DRY) principle. If we decide to rename this class at some point we won’t have to remember updating the constructor name in all of the classmethod factory functions.

class Pizza:
    def __init__(self, ingredients):
        self.ingredients = ingredients

    def __repr__(self):
        return f'Pizza({self.ingredients!r})'

    @classmethod
    def margherita(cls):
        return cls(['mozzarella', 'tomatoes'])

    @classmethod
    def prosciutto(cls):
        return cls(['mozzarella', 'tomatoes', 'ham'])





#!/usr/bin/env python
# coding: utf-8

# # TYPE OF OPERATORS
#  
#  An operator is a symbol that performs a certain operator between operands

# In[6]:


# 1) ARITHMETIC OPERATIONS 
a= 10
b= 20
print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a%b)
print(a**b)


# In[7]:


# 2) RELATIONAL/ COMPARISION OPERATORS
a = 10
b = 20

print(a == b)  #Equal to
print(a != b)  #Not equal to
print(a > b)   #Greater than
print(a < b)  #Less than
print(a >= b)  #Greater than or equal to
print(a <= b)  #Less than or equal to


# In[8]:


# 3) ASSIGNMENT OPERATOR
x = 10     #Assign value

x+=5   #Add and assign
x-=2   #Subtract and assign
x%=4   #Modulus and assign
x**=3   #Exponentiation and assign
x/=6    #Divide and assign
x//=2   #Floor division and assign
x*=6    #Multiply and assign


# In[9]:


# 4) LOGICAL OPERATOR

a = 10
b = 40

#AND OPERATOR
print((a<20) and (b>30))   #True if both conditions are True

#OR OPERATOR
print((a>20) or (b>20))    #True if at least one condition is True

#NOT OPERATOR
print(not(b>30))           #Reverse the result


# In[10]:


# 5) BITWISE OPERATOR
# Bitwise operators work on the binary format of numbers

a = 5      # (binary: 0101)
b = 3      # (binary: 0011)

# Bitwise AND
print(a & b)   # 0101 & 0011 = 0001 → Output: 1

# Bitwise OR
print(a | b)   # 0101 | 0011 = 0111 → Output: 7

# Bitwise XOR
print(a ^ b)   # 0101 ^ 0011 = 0110 → Output: 6

# Bitwise NOT
print(~a)      # ~0101 = 1010 → Output: -6 (Python mein complement hota hai: -(n+1))

# Left Shift
print(a << 1)  # 0101 << 1 = 1010 → Output: 10

# Right Shift
print(a >> 1)  # 0101 >> 1 = 0010 → Output: 2


# INPUT IN PYTHON
# 
# input()statement used to accept values fron user
# 
# input() # result is str
# 
# int(input()) # int
# 
# float(input()) #float

# # Data types

# In[11]:


# 1) STRING  " "
 # It is a data type that store a sequence of charactor
 # It is IMMUTABLE (VALUE CANNOT BE CHANGE AFTER CREATION)
 # strings are written inside quotes (' ' or " ").
 
str= "I AM IZRAM SANA AND I LOVE CODING"
print(str)

print(str.endswith("ER"))            # RETURNS true if string end with 'ER'
print(str.startswith("I"))
print(str.find("AM"))                # returns 1st index of 1st occurrer
print(str.replace("CODER","Human"))  # replace all occurrence of old
print(str.count("I"))                # counts the occurrence of sub str
 
    
print(len(str))     ##len function gives the length of string  

print("Hello"+"World ") # concatenation 


# # SLICING
# 
#  #Slicing means cutting or taking out a part of a string (or list).
#  
#  #You choose start, end, and sometimes a step.
# 
# 

# In[12]:


## string[start : end : step]

str= "Hi everyone these are my python notes "

print(str[1:9])
print(str[:5])
print(str[1:])
print(str[: :-1])
print(str[-3:-1])


# # INDEXING

# In[13]:


'''Indexing means accessing individual elements in a sequence (like a string, list, or tuple) using their position'''


#Indexing starts from 0 (zero-based).

#Python supports positive and negative indexing.
 
name = "Python"
print(name[0])   
print(name[1])  
print(name[-1])  


# In[14]:


# 2) LIST []

'''A built in data type that stores set of values .It can store element of different types (int,float,str,etc)'''

my_list = [10, 20, 30, 40]

# Add new element at the end
my_list.append(50)
print(my_list)  

# Insert element at a specific position
my_list.insert(2, 25)
print(my_list)  

# Remove an element
my_list.remove(30)
print(my_list)  

# Sort the list # sort in ascending order
my_list.sort()
print(my_list)  

# Reverse the list
my_list.reverse()
print(my_list)  

# Copy the list
new_list = my_list.copy()
print(new_list) 

# extend 
new_list.extend([60,70])
print(new_list)


# In[15]:


# 3) TUPLES  ()

# Features of Tuple:
#Ordered: Tuples maintain the order of elements.

#Allow duplicates: You can have repeated values.

#Immutable: Cannot change after creation (no append, remove, etc.).

#Faster than lists for reading data.

#Can hold different data types: integers, strings, floats, etc.

my_tuple = (1, 2, 3, 2, 2, 4)

print(my_tuple.count(2))  # returns the number of times a value appears. 
print(my_tuple.index(3))  # returns the index of the first occurrence of a value.



# In[16]:


# 4) DICTIONARY {}
# It is used to stored the data values in key:value pairs
#They are unordered  , mutable ,and don't allow duplicate keys

#Keys must be unique and immutable (strings, numbers, tuples, etc.).
#Values can be anything (strings, numbers, lists, even other dictionaries).

student = {
     "name":"Izram",
     "percentage":91,
     "marks":[98,96,95],
    } 
print(student)

# dictionary methods
print(student.keys())         #returns all keys 
print(student.values())       #returns all values
print(student.items())        #returns all(key,value)pairs
print(student.get("name"))    #returns the keys according to values

student.update({"class":10})   #insert the key value pair in the dictionary
print(student)

val=student.pop("class")      #remove specific key and returns its value
print(val)
print(student)

last_item=student.popitem()     #remove the last key value pair
print(student)

student.clear()                #remove all item from the dictionary
print(student)

# NESTED DICTIONARY
student2={"name":"Sana ",
         "subject":{
             "physics":93,
             "chemistry":90,
             "math":95 }}
print(student2)


# In[17]:


# 5) SETS ()  # THIS IS EMPTY SET ()

'''SET IS THE COLLECTION OF UNORDERED ITEMS. EACH ELEMENT IN THE SET MUST BE UNIQUE AND IMMUTABLE'''

set={1,2,2,3,4,4,"syed"}  # repeated element stored only once so it resolved to {1,2,3,4}
print(set)

#SET METHODS

set2={"syed",99,"iron man",21.5}
print(set2)

set2.add("thor")         #add an element
print(set2)

set2.remove("thor")      #remove an element
print(set2)

set.union(set2)          #combine both set values and return new

set.intersection(set2)   #combines common values and returns new  

set2.pop()               #remove a randon values
print(set2)

set2.clear()              #empties the set
print(set2)


# # CONDITIONAL STATEMENTS : IF , ELSE , ELIF

# In[18]:


#It control the flow of program based on condition(true /false)

# "IF" statement
age=18
if age>=18:
    print("You are an adult")
    
#"IF.... ELSE" statement
age=16
if age>=18:
    print("You can vote")
else:
    print("You are too young to vote")
    
    
# "IF...ELIF...ELSE" statement
marks= 75

if marks>=90:
    print("grade A")
elif marks>=80:
    print("grade B")
elif marks>=70:
    print("grade C")
else:
    print("grade D")

'''Conditions must be Boolean expressions (evaluate to True or False).

   Indentation (usually 4 spaces) is required in Python.

   You can nest if statements inside others.'''


# # LOOPS : for loops , while loops

# In[19]:


'''loops are used to repeat a blocks of code multiple times'''

# FOR LOOPS : Used to iterate over a sequence like (list, string, range)etc.


for i in range (5):    # range(start,stop,step)
    print(i)
    

fruits = ["apple","banana","orange","mango","kiwi"]    
for fruit in fruits:
    print(fruit)
    

# WHILE LOOPS : Repeats a block as long as condition is true

x = 1                # be careful to infinity loops 
while x<=5:
    print(x)
    x+=1
    


# # LOOPS CONTROL STATEMENTS : BREAK , CONTINUE , PASS

# In[20]:


'''Break : Used to exit the loop immediately when a condition is met'''

for i in range(10):
    if i==5:
        break
    print(i) 
    
'''Continue : Used to skip the current iteration and continue with the next one'''

for i in range(5):
    if i ==2:
        continue
    print(i,end=" ")  
    
    
'''Pass : Used as placeholder when you don't want to execute any code'''

for i in range (3):
    pass

''' ELSE WITH LOOPS : Both for and while can have an else clause. It runs only if the loop was not terminated by a break.'''

for i in range (3):
    print(i)
else:
    print("loop is completed")        


# # FUNCTIONS

# In[21]:


# A FUNCTION IS A GROUP OF CODE THAT DOES A SPECIFIC TASK

''' Use def() keyword to create a function'''

def greet():           #function define
    print("Hello!")
    
greet()                # function call


''' Function with Parameters : Sometimes we want to send values to a function. These values are called parameters.'''

def greet(name):
    print("Hello" , name)
  
greet("Izram")

'''Function with Default Value : If no value is given, a default value will be used.'''

def greet(name="friends"):
    print("Hi", name)
    
greet()
greet("Ali")

'''Function with Return Value : If we want the function to give back (return) a result, we use return.'''

def add(a,b):
    return a+b

result=add(4,5)
print(result)


'''*args (Multiple Values) : You can pass many values using *args.'''

def total(*numbers):
    print(sum(numbers))
    
total(1,2,3,4,5)


'''**kwargs (Many Key-Value Pairs)'''

def info(**data):
    print(data)
    
info(name="izram",age=21)    

''' Lambda Function (One-Line Shortcut) : Used for small, quick functions in one line.'''

square= lambda x:x*x
print(square(5))

'''Function Calling Itself (Recursion)'''

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

print(factorial(4)) 


# # FILE  HANDLING/  I / O 

# In[22]:


'''FILE HANDLING : Managing files using Python — like creating, opening, reading, writing, appending, closing, deleting.

                   It helps you work with files (like .txt) for storing and retrieving data permanently.'''


'''FILE I/O : Input = Reading data from a file.Output = Writing data to a file.

              File I/O is a part of File Handling.'''

# HOW TO CREATE A FILE:

#"w" mode → Creates a new file or overwrites if it exists

#"x" mode → Creates a new file only if it doesn’t exist

#"a" mode → Creates a file if it doesn’t exist, then adds to it

# FILE I/O FUNCTIONS:

# open("file", "r"):  Open a file for reading
# open("file", "w"):  Open for writing (overwrites)
# file.write(text):   Write text to a file
# file.read():        Read all content
# file.readline():    Read one line
# file.readlines():   Read all lines as list
# file.close():       Close the file (not needed with with)


# In[23]:


file = open("example.txt", "w")        # We create a file using "w" mode
file.write("This is a new file created by Python.")
file.close()


# In[24]:


import os      # check if  the file was created or not

print(os.path.exists("example.txt"))  # Output: True or False


# In[25]:


file=open("example.txt","w")
file.write("Hello, this is my code")
file.close


# In[26]:


file=open("example.txt","r")
print(file.read())
file.close


# In[27]:


file = open("example.txt", "a")
file.write("\nAdding new line")
file.close()


# # OOPS 

# In[28]:


# OBJECT ORIENTED PROGRAM OOPS

'''IT IS A WAY TO STRUCTURE THE CODE USING CLASS AND OBJECT. IT HELPS MAKE CODE ORGANIZED REUSEABLE AND EASIER TO MANAGE '''

# KEY CONCEPT OF OOPS
'''1) CLASS : BLUEPRINT/ TEMPLATE TO CREATE OBJECTS
   2) OBJECT: INTANCE OF A CLASS
   3)CONSTRUCTOR (__init__): special method to initialize objects
   4) SELF : REFERS TO THE CURRENT OBJECT
   5) METHOD : FUNCTION INSIDE A CLASS
   6) ATTRIBUTE : VARIABLE INSIDE A CLASS
   7)ENCAPSULATION: HIDE INTERNAL DATA
   8)INHERITANCE : ONE CLASS GETS FEATURES FROM ANOTHER CLASS
   9)POLYMORPHISM : ONE NAME/ THING MANY FORMS '''

# Define a class
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

    def show(self):
        print(f"Name: {self.name}, Grade: {self.grade}")

# Create an object
s1 = Student("Ali", "A")
s1.show()  


# In[29]:


# Inheritance Example
class Animal:
    def sound(self):
        print("Animal makes sound")

class Dog(Animal):
    def sound(self):
        print("Dog barks")

d = Dog()
d.sound() 


# In[30]:


#Encapsulation Example
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # private attribute

    def show_balance(self):
        print("Balance:", self.__balance)

    def deposit(self, amount):
        self.__balance += amount

account = BankAccount(1000)
account.deposit(500)
account.show_balance()  


# In[31]:


# Polymorphism Example
class Bird:
    def speak(self):
        print("Some sound")

class Parrot(Bird):
    def speak(self):
        print("Parrot says hello")

class Crow(Bird):
    def speak(self):
        print("Crow caws")

for bird in (Parrot(), Crow()):
    bird.speak()


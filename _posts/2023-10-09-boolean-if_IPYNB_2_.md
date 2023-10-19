---
title: Boolean Hacks
type: hacks
description: Boolean iF else
categories: ['C4.9']
courses: {'csp': {'week': 8}}
tags: ['pseudocode', 'python']
---

# Boolean

- A Boolean value is either true or false.
- A Boolean expression produces a Boolean value (true or false) when evaluated.

## <mark>Conditional ("if") statements</mark>

- Affect the sequential flow of control by executing different statements based on the value of a Boolean expression.


```python
IF (condition)
{
	<block of statements>
}

The code in <block of statements> is executed if the Boolean expression condition evaluates to true; no action is taken if condition evaluates to false.

IF (condition)
{
	<block of statements>
}
ELSE
{
	<second block of statements>
}

The code in the first <block of statements> is executed if the Boolean expression condition evaluates to true; otherwise, the code in <second block of statements> is executed.
```

<b>Example:</b> Calculate the sum of 2 numbers. If the sum is greater than 10, display 10; otherwise, display the sum.


```python
num1 = INPUT
num2 = INPUT
sum = num1 + num2
IF (sum > 10)
{
	DISPLAY (10)
}
ELSE
{
	DISPLAY (sum)
}
```

# Hack 1

- Add a variable that represents an age.

- Add an 'if' and 'print' function that says "You are an adult" if your age is greater than or equal to 18.

- Make a function that prints "You are a minor" with the else function.


```python
## YOUR CODE HERE
age=int(input())
if age>18:
    print("You are an adult")
else:
    print("You are a a minor")
```

## <mark><b>Relational operators:</b></mark> 
- Used to test the relationship between 2 variables, expressions, or values. These relational operators are used for comparisons and they evaluate to a Boolean value (true or false).

<b>Ex.</b> a == b evaluates to true if a and b are equal, otherwise evaluates to false

- a == b (equals)	
- a != b (not equal to)
- a > b (greater than)
- a < b (less than)
- a >= b (greater than or equal to)
- a <= b (less than or equal to)

<b>Example:</b> The legal age to work in California is 14 years old. How would we write a Boolean expression to check if someone is at least 14 years old?

age >= 14

<b>Example:</b> Write a Boolean expression to check if the average of height1, height2, and height3 is at least 65 inches.

(height1 + height2 + height3) / 3 >= 65

# Hack 2

- Make a variable called 'is_raining' and set it to 'True".

- Make an if statement that prints "Bring an umbrella!" if it is true

- Make an else statement that says "The weather is clear".


```python
## YOUR CODE HERE
is_raining=True
if is_raining==1:
    print("bring an umbrella")
else:
    print("the weather is clear")
```

## <mark><b>Logical operators:</b></mark>
Used to evaluate multiple conditions to produce a single Boolean value.

- <b>NOT</b>	evaluates to true if condition is false, otherwise evaluates to false
- <b>AND</b>	evaluates to true if both conditions are true, otherwise evaluates to false
- <b>OR</b>	evaluates to true if either condition is true or if both conditions are true, otherwise evaluates to false

<b>Example:</b> You win the game if you score at least 10 points and have 5 lives left or if you score at least 50 points and have more than 0 lives left. Write the Boolean expression for this scenario.

(score >= 10 AND lives == 5) OR (score == 50 AND lives > 0)

## <mark><b>Relational and logical operators:</b></mark>

<b>Example:</b> These expressions are all different but will produce the same result.

- age >= 16
- age > 16 OR age == 16
- NOT age < 16

# Hack 3

- Make a function to randomize numbers between 0 and 100 to be assigned to variables a and b using random.randint

- Print the values of the variables

- Print the relationship of the variables; a is more than, same as, or less than b


```python
## YOUR CODE HERE
import random
x=random.randint(0,100)
x1=random.randint(0,100)
if x==x1:
    print("a is the same as b", x, x1)
if x>=x1:
    print("a is greater then b", x,">", x1)
if x<=x1:
    print("b is greater then a", x,"<", x1)

```

    a is greater then b 30 > 1


## <b>Homework</b>

### Criteria for above 90%:
- Add more questions relating to Boolean rather than only one per topic (ideas: expand on conditional statements, relational/logical operators)
- Add a way to organize the user scores (possibly some kind of leaderboard, keep track of high score vs. current score, etc. Get creative!)
- Remember to test your code to make sure it functions correctly.


```python
# Import necessary modules
import getpass  # Module to get the user's name
import sys  # Module to access system-related information

# Function to ask a question and get a response
def question_with_response(prompt):
    # Print the question
    print("Question: " + prompt)
    # Get user input as the response
    msg = input()
    return msg

# Define the number of questions and initialize the correct answers counter
questions = 15
correct = 0
user_name = input("Enter your name: ")

# Question 1: Boolean Basics
response = question_with_response("What are the possible values of a Boolean variable?")
if response.lower() == "true or false":
    print("Correct!")
    correct += 1
else:
    print("Incorrect.")

# Question 2: Boolean Expressions
response = question_with_response("Why are Boolean expressions important in programming?")
if "control flow" in response.lower() and "decision making" in response.lower():
    print("Correct!")
    correct += 1
else:
    print("Incorrect.")

# Question 3: Conditional Statements
response = question_with_response("What is the purpose of conditional (if-else) statements in programming?")
if "execute different code" in response.lower() and "conditions met" in response.lower():
    print("Correct!")
    correct += 1
else:
    print("Incorrect.")

# Question 4: Relational Operators
response = question_with_response("Give an example of a relational operator in programming.")
if response.lower() == "==" or response.lower() == "<=" or response.lower() == ">=":
    print("Correct!")
    correct += 1
else:
    print("Incorrect.")

# Question 5: Logical Operators
response = question_with_response("Provide an example of a logical operator in programming.")
if response.lower() == "and" or response.lower() == "or" or response.lower() == "not":
    print("Correct!")
    correct += 1
else:
    print("Incorrect.")

# Question 6: Boolean Values
response = question_with_response("What is the result of the following Boolean expression: `(True and False) or (True or False)`?")
if response.lower() == "true":
    print("Correct!")
    correct += 1
else:
    print("Incorrect.")

# Question 7: Boolean Expressions
response = question_with_response("Explain the difference between the `and` and `or` logical operators in Python with examples.")
if "and operator" in response.lower() and "combines two conditions" in response.lower() and "both conditions true" in response.lower():
    print("Correct!")
    correct += 1
else:
    print("Incorrect.")

# Question 8: Conditional Statements
response = question_with_response("In Python, what is the difference between the `if`, `elif`, and `else` statements? Provide an example of how they are used together.")
if "if statement" in response.lower() and "checks a condition" in response.lower() and "elif statement" in response.lower() and "multiple conditions" in response.lower():
    print("Correct!")
    correct += 1
else:
    print("Incorrect.")

# Question 9: Relational Operators
response = question_with_response("Write a Python expression using a relational operator to check if a variable `x` is greater than or equal to 10.")
if response.lower() == "x >= 10":
    print("Correct!")
    correct += 1
else:
    print("Incorrect.")

# Question 10: Logical Operators
response = question_with_response("Explain the purpose of the `not` operator in Boolean logic. Provide an example to illustrate its usage.")
if "negates a condition" in response.lower() and "turns true into false and vice versa" in response.lower():
    print("Correct!")
    correct += 1
else:
    print("Incorrect.")

# Question 11: Boolean Logic
response = question_with_response("Simplify the following Boolean expression: `(True and False) or (not True)`")
if response.lower() == "False":
    print("Correct!")
    correct += 1
else:
    print("Incorrect.")

# Question 12: Conditional Statements
response = question_with_response("What is the difference between `=` and `==` in Python? Explain with examples.")
if "= is assignment operator" in response.lower() and "== is equality operator" in response.lower():
    print("Correct!")
    correct += 1
else:
    print("Incorrect.")

# Question 13: Logical Operators
response = question_with_response("Write a Python expression using the `or` operator to check if a number is either less than 0 or greater than 100.")
if response.lower() == "num < 0 or num > 100":
    print("Correct!")
    correct += 1
else:
    print("Incorrect.")

# Question 14: Boolean Expressions
response = question_with_response("How can parentheses be used to change the order of operations in a Boolean expression? Provide an example.")
if "change the evaluation order" in response.lower() and "control how expressions are evaluated" in response.lower():
    print("Correct!")
    correct += 1
else:
    print("Incorrect.")

# Question 15: Conditional Statements
response = question_with_response("Explain the concept of nested conditional statements with an example in Python.")
if "if statements inside other if statements" in response.lower() and "allows multiple branching levels" in response.lower():
    print("Correct!")
    correct += 1
else:
    print("Incorrect.")

# Display the final score
percentage_correct = (correct / questions) * 100
print(user_name + ", you scored " + str(correct) + "/" + str(questions) + " which is " + str(percentage_correct) + "%.")

```

    Question: What are the possible values of a Boolean variable?
    Incorrect.
    Question: Why are Boolean expressions important in programming?
    Incorrect.
    Question: What is the purpose of conditional (if-else) statements in programming?
    Incorrect.
    Question: Give an example of a relational operator in programming.
    Incorrect.
    Question: Provide an example of a logical operator in programming.
    Incorrect.
    Question: What is the result of the following Boolean expression: `(True and False) or (True or False)`?
    Incorrect.
    Question: Explain the difference between the `and` and `or` logical operators in Python with examples.
    Incorrect.
    Question: In Python, what is the difference between the `if`, `elif`, and `else` statements? Provide an example of how they are used together.
    Incorrect.
    Question: Write a Python expression using a relational operator to check if a variable `x` is greater than or equal to 10.
    Incorrect.
    Question: Explain the purpose of the `not` operator in Boolean logic. Provide an example to illustrate its usage.
    Incorrect.
    Question: Simplify the following Boolean expression: `(True and False) or (not True)`
    Incorrect.
    Question: What is the difference between `=` and `==` in Python? Explain with examples.
    Incorrect.
    Question: Write a Python expression using the `or` operator to check if a number is either less than 0 or greater than 100.
    Incorrect.
    Question: How can parentheses be used to change the order of operations in a Boolean expression? Provide an example.
    Incorrect.
    Question: Explain the concept of nested conditional statements with an example in Python.
    Incorrect.
    Saaras, you scored 0/15 which is 0.0%.


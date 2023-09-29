---
layout: post
title: Python IO
description: A cool introduction to python inputs/outputs
toc: True
comments: True
categories: ['5.A', 'C4.1']
courses: {'csse': {'week': 0}, 'csp': {'week': 1, 'categories': ['6.B']}, 'csa': {'week': 0}}
type: devops
---

## Python Input/Output:
- We were able to utilize our pre-requisite, knowledge of functions to create a set of questions and answers in a more optimal fashion where it is less lines of code and looks more aesthetic to the eye.




```python
import getpass, sys, os

def ask_question(question, correct_answer):
    user_response = input(f"Question: {question}\nYour Answer: ")
    if user_response.lower() == correct_answer.lower():
        print(f"Correct! {correct_answer} is the right answer.\n")
        return 1
    else:
        print(f"Sorry, {user_response} is incorrect. The correct answer is {correct_answer}.\n")
        return 0

def main():
    questions = [
        ("What is Python primarily used for?", "Python is primarily used for web development, data analysis, artificial intelligence, and automation."),
        ("How do you print 'Hello, World!' in Python?", "You can print 'Hello, World!' using the 'print()' function like this: 'print('Hello, World!')'."),
        ("What is a variable in Python?", "A variable is a name used to store data in Python. It can hold different types of data, such as numbers, text, or lists."),
    ]

    print(f"Hello, {getpass.getuser()} running {sys.executable}")
    print(f"You will be asked {len(questions)} questions.")

    input("Press Enter to start the test...")

    correct_count = sum(ask_question(q, a) for q, a in questions)

    print(f"{getpass.getuser()}, you scored {correct_count}/{len(questions)}.")

if __name__ == "__main__":
    main()

```

    Hello, kodal running C:\Users\kodal\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe
    You will be asked 3 questions.


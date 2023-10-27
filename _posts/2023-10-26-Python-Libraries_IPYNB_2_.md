---
toc: True
comments: True
layout: post
title: Student Lesson Python Libraries
description: To teach the class how to use public Python libraries around the internet
type: hacks
courses: {'csp': {'week': 10}}
---

### What is a Library?
Essentially a list of pre-written code that you can use to streamline and clean up your program.

Libraries can help simplify complex programs

APIS are specifications for how the procedures in a library behave, and how they can be used 

Documentations for an API/library is necessary in understanding the behaviors provided by the API/library and how to use them

Libraries that we will go over: Requests, Pillow, Pandas, Numpy, Scikit-Learn, TensorFlow, matplotlib.


### Required Installations
Please run the following commands in your vscode terminal in order to continue the lesson
- pip install numpy && pip install matplotlib && pip install scikit-learn && pip install pillow && pip install pandas && pip install tensorflow && pip install requests

### Images using requests and pillow libraries
'Requests' is focused on handling HTTP requests and web data while 'Pillow' is designed for data manipulation and analysis
It's common to see them used together in data-related assignments where data is fetched by HTTP requests using Requests and then processed and analyzed with Pandas.

Here's an example:


```python
pip install numpy
```

    Collecting numpy
      Obtaining dependency information for numpy from https://files.pythonhosted.org/packages/82/0f/3f712cd84371636c5375d2dd70e7514d264cec6bdfc3d7997a4236e9f948/numpy-1.26.1-cp311-cp311-win_amd64.whl.metadata
      Downloading numpy-1.26.1-cp311-cp311-win_amd64.whl.metadata (61 kB)
         ---------------------------------------- 0.0/61.2 kB ? eta -:--:--
         ---------------------------------------- 61.2/61.2 kB 3.4 MB/s eta 0:00:00
    Downloading numpy-1.26.1-cp311-cp311-win_amd64.whl (15.8 MB)
       ---------------------------------------- 0.0/15.8 MB ? eta -:--:--
       -- ------------------------------------- 0.9/15.8 MB 27.7 MB/s eta 0:00:01
       ----- ---------------------------------- 2.0/15.8 MB 25.2 MB/s eta 0:00:01
       ------ --------------------------------- 2.6/15.8 MB 23.4 MB/s eta 0:00:01
       ------- -------------------------------- 3.0/15.8 MB 19.0 MB/s eta 0:00:01
       -------- ------------------------------- 3.2/15.8 MB 16.9 MB/s eta 0:00:01
       --------- ------------------------------ 3.9/15.8 MB 16.6 MB/s eta 0:00:01
       ------------ --------------------------- 5.0/15.8 MB 18.9 MB/s eta 0:00:01
       --------------- ------------------------ 6.3/15.8 MB 20.1 MB/s eta 0:00:01
       ------------------ --------------------- 7.2/15.8 MB 20.1 MB/s eta 0:00:01
       --------------------- ------------------ 8.5/15.8 MB 20.9 MB/s eta 0:00:01
       ------------------------ --------------- 9.7/15.8 MB 22.1 MB/s eta 0:00:01
       -------------------------- ------------- 10.5/15.8 MB 21.1 MB/s eta 0:00:01
       ----------------------------- ---------- 11.5/15.8 MB 21.8 MB/s eta 0:00:01
       ------------------------------- -------- 12.5/15.8 MB 21.8 MB/s eta 0:00:01
       --------------------------------- ------ 13.1/15.8 MB 23.4 MB/s eta 0:00:01
       ------------------------------------ --- 14.4/15.8 MB 25.2 MB/s eta 0:00:01
       ---------------------------------------  15.7/15.8 MB 25.1 MB/s eta 0:00:01
       ---------------------------------------  15.8/15.8 MB 24.2 MB/s eta 0:00:01
       ---------------------------------------- 15.8/15.8 MB 22.6 MB/s eta 0:00:00
    Installing collected packages: numpy
    Successfully installed numpy-1.26.1
    Note: you may need to restart the kernel to use updated packages.


    
    [notice] A new release of pip is available: 23.2.1 -> 23.3.1
    [notice] To update, run: python.exe -m pip install --upgrade pip



```python
import requests
from PIL import Image
from io import BytesIO

# Step 1: Download an image using Requests
image_url = "https://example.com/path/to/your/image.jpg"  # Replace with the actual URL of the image you want to download
response = requests.get(image_url)

if response.status_code == 200:
    # Step 2: Process the downloaded image using Pillow
    image_data = BytesIO(response.content)  # Create an in-memopry binary stream from the response content
    img = Image.open(image_data)  # Open the image using Pillow

    # Perform image processing tasks here, like resizing or applying filters
    img = img.resize((x, y))  # Resize the image and replace x,y with desired amounts

    # Step 3: Save the processed image using Pillow
    img.save("processed_image.jpg")  # Save the processed image to a file

    print("Image downloaded, processed, and saved.")
else:
    print(f"Failed to download image. Status code: {response.status_code}")

```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    c:\Users\kodal\student-public\_notebooks\2023-10-26-Python-Libraries.ipynb Cell 5 line 2
          <a href='vscode-notebook-cell:/c%3A/Users/kodal/student-public/_notebooks/2023-10-26-Python-Libraries.ipynb#W4sZmlsZQ%3D%3D?line=0'>1</a> import requests
    ----> <a href='vscode-notebook-cell:/c%3A/Users/kodal/student-public/_notebooks/2023-10-26-Python-Libraries.ipynb#W4sZmlsZQ%3D%3D?line=1'>2</a> from PIL import Image
          <a href='vscode-notebook-cell:/c%3A/Users/kodal/student-public/_notebooks/2023-10-26-Python-Libraries.ipynb#W4sZmlsZQ%3D%3D?line=2'>3</a> from io import BytesIO
          <a href='vscode-notebook-cell:/c%3A/Users/kodal/student-public/_notebooks/2023-10-26-Python-Libraries.ipynb#W4sZmlsZQ%3D%3D?line=4'>5</a> # Step 1: Download an image using Requests


    ModuleNotFoundError: No module named 'PIL'


In this code, we use the Requests library to download an image from a URL and then if the download is successful the HTTP status code 200 will pop up, and from there we create an in-memory binary stream (BytesIO) from the response content. We then use the Pillow library to open the image, make any necessary changes, and save the processed image to a file.

Here's a step by step tutorial on how we wrote this code: 
1)We started by importing the necessary libraries, which were Requests, Pillow, and io.

2)Download the Image

3)Use the Requests library to send an HTTP GET request to the URL to download the image.
Check the response status code to make sure the download goes through(status code 200).

4)If the download is successful, create an in-memory binary stream (BytesIO) from the response content.
Process the Image:

5)Utilize the Pillow library to open the image from the binary stream.
Change photo to desired preference(ie: size)
Save the Processed Image:

6)Save the processed image to a file using Pillow. Choose a filename and file format for the saved image.




### Hack 1

Write a Python code that accomplishes the following tasks:

Downloads an image from a specified URL using the Requests library.
Processes the downloaded image (like resizing) using the Pillow library.
Save the processed image to a file.



```python
import requests
from PIL import Image
from io import BytesIO

# Step 1: Download an image using Requests
image_url = "https://media.istockphoto.com/id/517188688/photo/mountain-landscape.jpg?s=612x612&w=0&k=20&c=A63koPKaCyIwQWOTFBRWXj_PwCrR4cEoOw2S9Q7yVl8="  # Replace with the actual URL of the image you want to download
response = requests.get(image_url)

if response.status_code == 200:
    # Step 2: Process the downloaded image using Pillow
    image_data = BytesIO(response.content)  # Create an in-memopry binary stream from the response content
    img = Image.open(image_data)  # Open the image using Pillow

    # Perform image processing tasks here, like resizing or applying filters
    img = img.resize((x, y))  # Resize the image and replace x,y with desired amounts

    # Step 3: Save the processed image using Pillow
    img.save("processed_image.jpg")  # Save the processed image to a file

    print("Image downloaded, processed, and saved.")
else:
    print(f"Failed to download image. Status code: {response.status_code}")

```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    c:\Users\kodal\student-public\_notebooks\2023-10-26-Python-Libraries.ipynb Cell 9 line 2
          <a href='vscode-notebook-cell:/c%3A/Users/kodal/student-public/_notebooks/2023-10-26-Python-Libraries.ipynb#X11sZmlsZQ%3D%3D?line=0'>1</a> import requests
    ----> <a href='vscode-notebook-cell:/c%3A/Users/kodal/student-public/_notebooks/2023-10-26-Python-Libraries.ipynb#X11sZmlsZQ%3D%3D?line=1'>2</a> from PIL import Image
          <a href='vscode-notebook-cell:/c%3A/Users/kodal/student-public/_notebooks/2023-10-26-Python-Libraries.ipynb#X11sZmlsZQ%3D%3D?line=2'>3</a> from io import BytesIO
          <a href='vscode-notebook-cell:/c%3A/Users/kodal/student-public/_notebooks/2023-10-26-Python-Libraries.ipynb#X11sZmlsZQ%3D%3D?line=4'>5</a> # Step 1: Download an image using Requests


    ModuleNotFoundError: No module named 'PIL'


### Math Operations With Python Libraries
Numpy(Numerical Python) is used for numerical and scientific computing. It provides tools for handling large sets of numbers, such as data tables and arrays. Numpy makes it easier and more efficient to do mathematical tasks. 

The Matplotlib library lets you create a visual representation of your data (graphs, charts, and etc.)

### Example Sine Graph
Uses numpy and matplotlib libaries


```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data with NumPy
x = np.linspace(0, 2 * np.pi, 100) 
# Create an array of values from 0 to 2*pi
# 100 is included to have 100 points distributed between 0 and 2π to make graph smoother
y = np.sin(x)
# Compute the sine of each value

# Create a simple line plot using Matplotlib
plt.plot(x, y, label='Sine Function', color='blue', linestyle='-')  # Create the plot
plt.title('Sine Function')  # Set the title
plt.xlabel('x')  # Label for the x-axis
plt.ylabel('sin(x)')  # Label for the y-axis
plt.grid(True)  # Display a grid
plt.legend()  # Show the legend
plt.show()  # Display the plot

```


    
![png](output_11_0.png)
    



```python
pip install matplotlib
```

    Note: you may need to restart the kernel to use updated packages.Collecting matplotlib
      Obtaining dependency information for matplotlib from https://files.pythonhosted.org/packages/40/d9/c1784db9db0d484c8e5deeafbaac0d6ed66e165c6eb4a74fb43a5fa947d9/matplotlib-3.8.0-cp311-cp311-win_amd64.whl.metadata
      Downloading matplotlib-3.8.0-cp311-cp311-win_amd64.whl.metadata (5.9 kB)
    Collecting contourpy>=1.0.1 (from matplotlib)
      Obtaining dependency information for contourpy>=1.0.1 from https://files.pythonhosted.org/packages/e5/76/94bc17eb868f8c7397f8fdfdeae7661c1b9a35f3a7219da308596e8c252a/contourpy-1.1.1-cp311-cp311-win_amd64.whl.metadata
      Downloading contourpy-1.1.1-cp311-cp311-win_amd64.whl.metadata (5.9 kB)
    Collecting cycler>=0.10 (from matplotlib)
      Obtaining dependency information for cycler>=0.10 from https://files.pythonhosted.org/packages/e7/05/c19819d5e3d95294a6f5947fb9b9629efb316b96de511b418c53d245aae6/cycler-0.12.1-py3-none-any.whl.metadata
      Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
    Collecting fonttools>=4.22.0 (from matplotlib)
      Obtaining dependency information for fonttools>=4.22.0 from https://files.pythonhosted.org/packages/ae/f6/724d2d236797ea7479a5a7ec8e69c2bee60cad70273cf25078810415ae2d/fonttools-4.43.1-cp311-cp311-win_amd64.whl.metadata
      Downloading fonttools-4.43.1-cp311-cp311-win_amd64.whl.metadata (155 kB)
         ---------------------------------------- 0.0/155.5 kB ? eta -:--:--
         --------------------------- ---------- 112.6/155.5 kB 3.3 MB/s eta 0:00:01
         -------------------------------------- 155.5/155.5 kB 3.2 MB/s eta 0:00:00
    Collecting kiwisolver>=1.0.1 (from matplotlib)
      Obtaining dependency information for kiwisolver>=1.0.1 from https://files.pythonhosted.org/packages/1e/37/d3c2d4ba2719059a0f12730947bbe1ad5ee8bff89e8c35319dcb2c9ddb4c/kiwisolver-1.4.5-cp311-cp311-win_amd64.whl.metadata
      Downloading kiwisolver-1.4.5-cp311-cp311-win_amd64.whl.metadata (6.5 kB)
    Requirement already satisfied: numpy<2,>=1.21 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from matplotlib) (1.26.1)
    Requirement already satisfied: packaging>=20.0 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from matplotlib) (23.1)
    Collecting pillow>=6.2.0 (from matplotlib)
      Obtaining dependency information for pillow>=6.2.0 from https://files.pythonhosted.org/packages/b1/38/31def4109acd4db10672df6f806b175c0d21458f845ddc0890e43238ba7c/Pillow-10.1.0-cp311-cp311-win_amd64.whl.metadata
      Downloading Pillow-10.1.0-cp311-cp311-win_amd64.whl.metadata (9.6 kB)
    Collecting pyparsing>=2.3.1 (from matplotlib)
      Obtaining dependency information for pyparsing>=2.3.1 from https://files.pythonhosted.org/packages/39/92/8486ede85fcc088f1b3dba4ce92dd29d126fd96b0008ea213167940a2475/pyparsing-3.1.1-py3-none-any.whl.metadata
      Using cached pyparsing-3.1.1-py3-none-any.whl.metadata (5.1 kB)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from matplotlib) (2.8.2)
    Requirement already satisfied: six>=1.5 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)
    Downloading matplotlib-3.8.0-cp311-cp311-win_amd64.whl (7.6 MB)
       ---------------------------------------- 0.0/7.6 MB ? eta -:--:--
       --- ------------------------------------ 0.6/7.6 MB 19.5 MB/s eta 0:00:01
       ---------- ----------------------------- 2.0/7.6 MB 24.9 MB/s eta 0:00:01
       ----------------- ---------------------- 3.4/7.6 MB 26.8 MB/s eta 0:00:01
       ---------------------- ----------------- 4.3/7.6 MB 27.3 MB/s eta 0:00:01
       --------------------------- ------------ 5.2/7.6 MB 23.9 MB/s eta 0:00:01
       ----------------------------------- ---- 6.7/7.6 MB 25.3 MB/s eta 0:00:01
       ---------------------------------------  7.6/7.6 MB 25.7 MB/s eta 0:00:01
       ---------------------------------------- 7.6/7.6 MB 23.2 MB/s eta 0:00:00
    Downloading contourpy-1.1.1-cp311-cp311-win_amd64.whl (480 kB)
       ---------------------------------------- 0.0/480.5 kB ? eta -:--:--
       --------------------------------------- 480.5/480.5 kB 15.2 MB/s eta 0:00:00
    Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
    Downloading fonttools-4.43.1-cp311-cp311-win_amd64.whl (2.1 MB)
       ---------------------------------------- 0.0/2.1 MB ? eta -:--:--
       --------------------- ------------------ 1.2/2.1 MB 36.5 MB/s eta 0:00:01
       ---------------------------------------  2.1/2.1 MB 27.3 MB/s eta 0:00:01
       ---------------------------------------- 2.1/2.1 MB 22.7 MB/s eta 0:00:00
    Using cached kiwisolver-1.4.5-cp311-cp311-win_amd64.whl (56 kB)
    Downloading Pillow-10.1.0-cp311-cp311-win_amd64.whl (2.6 MB)
       ---------------------------------------- 0.0/2.6 MB ? eta -:--:--
       -------------------- ------------------- 1.3/2.6 MB 41.3 MB/s eta 0:00:01
       -------------------------------- ------- 2.1/2.6 MB 27.3 MB/s eta 0:00:01
       ---------------------------------------  2.6/2.6 MB 23.6 MB/s eta 0:00:01
       ---------------------------------------- 2.6/2.6 MB 20.8 MB/s eta 0:00:00
    Using cached pyparsing-3.1.1-py3-none-any.whl (103 kB)
    Installing collected packages: pyparsing, pillow, kiwisolver, fonttools, cycler, contourpy, matplotlib
    Successfully installed contourpy-1.1.1 cycler-0.12.1 fonttools-4.43.1 kiwisolver-1.4.5 matplotlib-3.8.0 pillow-10.1.0 pyparsing-3.1.1
    


    
    [notice] A new release of pip is available: 23.2.1 -> 23.3.1
    [notice] To update, run: python.exe -m pip install --upgrade pip


### Hack 2
Using the data from the numpy library, create a visual graph using different matplotlib functions.


```python
import numpy as np
import matplotlib.pyplot as plt
# Generate data for two lines
x = np.linspace(0, 10, 50)  # Create an array of values from 0 to 10
y1 = 2 * x + 1  # Set of data poits

# Create and display a plot using Matplotlib

# your code here

```


      Cell In[15], line 3
        pip install sklearn
            ^
    SyntaxError: invalid syntax




```python
pip install tensorflow
```

    Collecting tensorflow
      Obtaining dependency information for tensorflow from https://files.pythonhosted.org/packages/80/6f/57d36f6507e432d7fc1956b2e9e8530c5c2d2bfcd8821bcbfae271cd6688/tensorflow-2.14.0-cp311-cp311-win_amd64.whl.metadata
      Using cached tensorflow-2.14.0-cp311-cp311-win_amd64.whl.metadata (3.3 kB)
    Collecting tensorflow-intel==2.14.0 (from tensorflow)
      Obtaining dependency information for tensorflow-intel==2.14.0 from https://files.pythonhosted.org/packages/ad/6e/1bfe367855dd87467564f7bf9fa14f3b17889988e79598bc37bf18f5ffb6/tensorflow_intel-2.14.0-cp311-cp311-win_amd64.whl.metadata
      Using cached tensorflow_intel-2.14.0-cp311-cp311-win_amd64.whl.metadata (4.8 kB)
    Requirement already satisfied: absl-py>=1.0.0 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (2.0.0)
    Requirement already satisfied: astunparse>=1.6.0 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (1.6.3)
    Requirement already satisfied: flatbuffers>=23.5.26 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (23.5.26)
    Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (0.5.4)
    Requirement already satisfied: google-pasta>=0.1.1 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (0.2.0)
    Requirement already satisfied: h5py>=2.9.0 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (3.10.0)
    Requirement already satisfied: libclang>=13.0.0 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (16.0.6)
    Requirement already satisfied: ml-dtypes==0.2.0 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (0.2.0)
    Requirement already satisfied: numpy>=1.23.5 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (1.26.1)
    Requirement already satisfied: opt-einsum>=2.3.2 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (3.3.0)
    Requirement already satisfied: packaging in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (23.1)
    Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (4.24.4)
    Requirement already satisfied: setuptools in c:\program files\windowsapps\pythonsoftwarefoundation.python.3.11_3.11.1264.0_x64__qbz5n2kfra8p0\lib\site-packages (from tensorflow-intel==2.14.0->tensorflow) (65.5.0)
    Requirement already satisfied: six>=1.12.0 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (1.16.0)
    Requirement already satisfied: termcolor>=1.1.0 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (2.3.0)
    Requirement already satisfied: typing-extensions>=3.6.6 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (4.8.0)
    Requirement already satisfied: wrapt<1.15,>=1.11.0 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (1.14.1)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (0.31.0)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (1.59.0)
    Requirement already satisfied: tensorboard<2.15,>=2.14 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (2.14.1)
    Requirement already satisfied: tensorflow-estimator<2.15,>=2.14.0 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (2.14.0)
    Requirement already satisfied: keras<2.15,>=2.14.0 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorflow-intel==2.14.0->tensorflow) (2.14.0)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from astunparse>=1.6.0->tensorflow-intel==2.14.0->tensorflow) (0.41.2)
    Requirement already satisfied: google-auth<3,>=1.6.3 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2.23.3)
    Requirement already satisfied: google-auth-oauthlib<1.1,>=0.5 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (1.0.0)
    Requirement already satisfied: markdown>=2.6.8 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (3.5)
    Requirement already satisfied: requests<3,>=2.21.0 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2.31.0)
    Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (0.7.2)
    Requirement already satisfied: werkzeug>=1.0.1 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (3.0.1)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (5.3.2)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (0.3.0)
    Requirement already satisfied: rsa<5,>=3.1.4 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (4.9)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from google-auth-oauthlib<1.1,>=0.5->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (1.3.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from requests<3,>=2.21.0->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (3.2.0)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from requests<3,>=2.21.0->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from requests<3,>=2.21.0->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from requests<3,>=2.21.0->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2023.7.22)
    Requirement already satisfied: MarkupSafe>=2.1.1 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from werkzeug>=1.0.1->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2.1.3)
    Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (0.5.0)
    Requirement already satisfied: oauthlib>=3.0.0 in c:\users\kodal\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<1.1,>=0.5->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (3.2.2)
    Using cached tensorflow-2.14.0-cp311-cp311-win_amd64.whl (2.1 kB)
    Using cached tensorflow_intel-2.14.0-cp311-cp311-win_amd64.whl (284.2 MB)
    Installing collected packages: tensorflow-intel, tensorflow
    Note: you may need to restart the kernel to use updated packages.


    ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: 'C:\\Users\\kodal\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\tensorflow\\include\\external\\com_github_grpc_grpc\\src\\core\\ext\\filters\\client_channel\\lb_policy\\grpclb\\client_load_reporting_filter.h'
    HINT: This error might have occurred since this system does not have Windows Long Path support enabled. You can find information on how to enable this at https://pip.pypa.io/warnings/enable-long-paths
    
    
    [notice] A new release of pip is available: 23.2.1 -> 23.3.1
    [notice] To update, run: python.exe -m pip install --upgrade pip


Tensor Flow is used in deep learning and neural networks, while scikit-learn is used for typical machine learning tasks. When used together, they can tackle machine learning projects. In the code below, Tensor Flow is used for model creation and training. Scikit-learn is used for data-processing and model evaluation.

## Pip install tensorflow scikit-learn


```python
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 1)  # Feature
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)  # Target variable with noise
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create a simple linear regression model using TensorFlow and Keras
model = keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(1)
])
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Calculate the Mean Squared Error on the test set
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    c:\Users\kodal\student-public\_notebooks\2023-10-26-Python-Libraries.ipynb Cell 20 line 2
          <a href='vscode-notebook-cell:/c%3A/Users/kodal/student-public/_notebooks/2023-10-26-Python-Libraries.ipynb#X22sZmlsZQ%3D%3D?line=0'>1</a> import numpy as np
    ----> <a href='vscode-notebook-cell:/c%3A/Users/kodal/student-public/_notebooks/2023-10-26-Python-Libraries.ipynb#X22sZmlsZQ%3D%3D?line=1'>2</a> import tensorflow as tf
          <a href='vscode-notebook-cell:/c%3A/Users/kodal/student-public/_notebooks/2023-10-26-Python-Libraries.ipynb#X22sZmlsZQ%3D%3D?line=2'>3</a> from sklearn.model_selection import train_test_split
          <a href='vscode-notebook-cell:/c%3A/Users/kodal/student-public/_notebooks/2023-10-26-Python-Libraries.ipynb#X22sZmlsZQ%3D%3D?line=3'>4</a> from sklearn.metrics import mean_squared_error


    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\tensorflow\__init__.py:38
         35 import sys as _sys
         36 import typing as _typing
    ---> 38 from tensorflow.python.tools import module_util as _module_util
         39 from tensorflow.python.util.lazy_loader import LazyLoader as _LazyLoader
         41 # Make sure code inside the TensorFlow codebase can use tf2.enabled() at import.


    ModuleNotFoundError: No module named 'tensorflow.python'


A decrease in loss and time metrics (ms/epoch and ms/step) shows the efficiency increases as the training epochs increases

## Hack
fill in the missing code to match the custom data set


```python
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
# Generate a custom dataset (replace this with your data loading code)
# Synthetic data: House prices based on number of bedrooms and square footage
np.random.seed(0)
num_samples = 100
bedrooms = np.random.randint(1, 5, num_samples)
square_footage = np.random.randint(1000, 2500, num_samples)
house_prices = 100000 + 50000 * bedrooms + 100 * square_footage + 10000 * np.random.randn(num_samples)
# Combine features (bedrooms and square footage) into one array
X = np.column_stack((bedrooms, square_footage))
y = house_prices.reshape(-1, 1)
# Split the data into training and testing sets

# Standardize the features

# Create a regression model using TensorFlow and Keras

    # Input shape adjusted to the number of features
     # Output layer for regression

# Compile the model for regression
  # Using MSE as the loss function
# Train the model

# Make predictions on the test set

# Calculate the Mean Squared Error on the test set

```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    c:\Users\kodal\student-public\_notebooks\2023-10-26-Python-Libraries.ipynb Cell 23 line 2
          <a href='vscode-notebook-cell:/c%3A/Users/kodal/student-public/_notebooks/2023-10-26-Python-Libraries.ipynb#X31sZmlsZQ%3D%3D?line=0'>1</a> import numpy as np
    ----> <a href='vscode-notebook-cell:/c%3A/Users/kodal/student-public/_notebooks/2023-10-26-Python-Libraries.ipynb#X31sZmlsZQ%3D%3D?line=1'>2</a> import tensorflow as tf
          <a href='vscode-notebook-cell:/c%3A/Users/kodal/student-public/_notebooks/2023-10-26-Python-Libraries.ipynb#X31sZmlsZQ%3D%3D?line=2'>3</a> from sklearn.model_selection import train_test_split
          <a href='vscode-notebook-cell:/c%3A/Users/kodal/student-public/_notebooks/2023-10-26-Python-Libraries.ipynb#X31sZmlsZQ%3D%3D?line=3'>4</a> from sklearn.metrics import mean_squared_error


    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\tensorflow\__init__.py:38
         35 import sys as _sys
         36 import typing as _typing
    ---> 38 from tensorflow.python.tools import module_util as _module_util
         39 from tensorflow.python.util.lazy_loader import LazyLoader as _LazyLoader
         41 # Make sure code inside the TensorFlow codebase can use tf2.enabled() at import.


    ModuleNotFoundError: No module named 'tensorflow.python'


## HOMEWORK 1

Create a GPA calculator using Pandas and Matplot libraries and make:
1) A dataframe
2) A specified dictionary
3) and a print function that outputs the final GPA

Extra points can be earned with creativity.


```python
# your code here
```

## HOMEWORK 2

Import and use the "random" library to generate 50 different points from the range 0-100, then display the randomized data using a scatter plot.

Extra points can be earned with creativity.


```python
# your code here
```

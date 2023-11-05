---
toc: True
comments: True
layout: post
title: Student Lesson Python Libraries
description: To teach the class how to use public Python libraries around the internet
type: hacks
courses: {'csp': {'week': 10}}
---

```python
pip install tensorflow
```

    Collecting tensorflow
      Obtaining dependency information for tensorflow from https://files.pythonhosted.org/packages/80/6f/57d36f6507e432d7fc1956b2e9e8530c5c2d2bfcd8821bcbfae271cd6688/tensorflow-2.14.0-cp311-cp311-win_amd64.whl.metadata
      Using cached tensorflow-2.14.0-cp311-cp311-win_amd64.whl.metadata (3.3 kB)
    Collecting tensorflow-intel==2.14.0 (from tensorflow)
      Obtaining dependency information for tensorflow-intel==2.14.0 from https://files.pythonhosted.org/packages/ad/6e/1bfe367855dd87467564f7bf9fa14f3b17889988e79598bc37bf18f5ffb6/tensorflow_intel-2.14.0-cp311-cp311-win_amd64.whl.metadata
      Using cached tensorflow_intel-2.14.0-cp311-cp311-win_amd64.whl.metadata (4.8 kB)
    Collecting absl-py>=1.0.0 (from tensorflow-intel==2.14.0->tensorflow)
      Obtaining dependency information for absl-py>=1.0.0 from https://files.pythonhosted.org/packages/01/e4/dc0a1dcc4e74e08d7abedab278c795eef54a224363bb18f5692f416d834f/absl_py-2.0.0-py3-none-any.whl.metadata
      Using cached absl_py-2.0.0-py3-none-any.whl.metadata (2.3 kB)
    Collecting astunparse>=1.6.0 (from tensorflow-intel==2.14.0->tensorflow)
      Using cached astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
    Collecting flatbuffers>=23.5.26 (from tensorflow-intel==2.14.0->tensorflow)
      Obtaining dependency information for flatbuffers>=23.5.26 from https://files.pythonhosted.org/packages/6f/12/d5c79ee252793ffe845d58a913197bfa02ae9a0b5c9bc3dc4b58d477b9e7/flatbuffers-23.5.26-py2.py3-none-any.whl.metadata
      Using cached flatbuffers-23.5.26-py2.py3-none-any.whl.metadata (850 bytes)
    Collecting gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 (from tensorflow-intel==2.14.0->tensorflow)
      Using cached gast-0.5.4-py3-none-any.whl (19 kB)
    Collecting google-pasta>=0.1.1 (from tensorflow-intel==2.14.0->tensorflow)
      Using cached google_pasta-0.2.0-py3-none-any.whl (57 kB)
    Requirement already satisfied: h5py>=2.9.0 in c:\users\kodal\anaconda3\lib\site-packages (from tensorflow-intel==2.14.0->tensorflow) (3.7.0)
    Collecting libclang>=13.0.0 (from tensorflow-intel==2.14.0->tensorflow)
      Obtaining dependency information for libclang>=13.0.0 from https://files.pythonhosted.org/packages/02/8c/dc970bc00867fe290e8c8a7befa1635af716a9ebdfe3fb9dce0ca4b522ce/libclang-16.0.6-py2.py3-none-win_amd64.whl.metadata
      Using cached libclang-16.0.6-py2.py3-none-win_amd64.whl.metadata (5.3 kB)
    Collecting ml-dtypes==0.2.0 (from tensorflow-intel==2.14.0->tensorflow)
      Obtaining dependency information for ml-dtypes==0.2.0 from https://files.pythonhosted.org/packages/08/89/c727fde1a3d12586e0b8c01abf53754707d76beaa9987640e70807d4545f/ml_dtypes-0.2.0-cp311-cp311-win_amd64.whl.metadata
      Using cached ml_dtypes-0.2.0-cp311-cp311-win_amd64.whl.metadata (20 kB)
    Requirement already satisfied: numpy>=1.23.5 in c:\users\kodal\anaconda3\lib\site-packages (from tensorflow-intel==2.14.0->tensorflow) (1.24.3)
    Collecting opt-einsum>=2.3.2 (from tensorflow-intel==2.14.0->tensorflow)
      Using cached opt_einsum-3.3.0-py3-none-any.whl (65 kB)
    Requirement already satisfied: packaging in c:\users\kodal\anaconda3\lib\site-packages (from tensorflow-intel==2.14.0->tensorflow) (23.0)
    Collecting protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 (from tensorflow-intel==2.14.0->tensorflow)
      Obtaining dependency information for protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 from https://files.pythonhosted.org/packages/c2/59/f89c04923d68595d359f4cd7adbbdf5e5d791257945f8873d88b2fd1f979/protobuf-4.24.4-cp310-abi3-win_amd64.whl.metadata
      Using cached protobuf-4.24.4-cp310-abi3-win_amd64.whl.metadata (540 bytes)
    Requirement already satisfied: setuptools in c:\users\kodal\anaconda3\lib\site-packages (from tensorflow-intel==2.14.0->tensorflow) (68.0.0)
    Requirement already satisfied: six>=1.12.0 in c:\users\kodal\anaconda3\lib\site-packages (from tensorflow-intel==2.14.0->tensorflow) (1.16.0)
    Collecting termcolor>=1.1.0 (from tensorflow-intel==2.14.0->tensorflow)
      Using cached termcolor-2.3.0-py3-none-any.whl (6.9 kB)
    Requirement already satisfied: typing-extensions>=3.6.6 in c:\users\kodal\anaconda3\lib\site-packages (from tensorflow-intel==2.14.0->tensorflow) (4.7.1)
    Requirement already satisfied: wrapt<1.15,>=1.11.0 in c:\users\kodal\anaconda3\lib\site-packages (from tensorflow-intel==2.14.0->tensorflow) (1.14.1)
    Collecting tensorflow-io-gcs-filesystem>=0.23.1 (from tensorflow-intel==2.14.0->tensorflow)
      Using cached tensorflow_io_gcs_filesystem-0.31.0-cp311-cp311-win_amd64.whl (1.5 MB)
    Collecting grpcio<2.0,>=1.24.3 (from tensorflow-intel==2.14.0->tensorflow)
      Obtaining dependency information for grpcio<2.0,>=1.24.3 from https://files.pythonhosted.org/packages/13/ee/b698a4d0eeae2d58d9906f20959b58a00622180ff8bf4ba71f71aad3c7f8/grpcio-1.59.2-cp311-cp311-win_amd64.whl.metadata
      Downloading grpcio-1.59.2-cp311-cp311-win_amd64.whl.metadata (4.2 kB)
    Collecting tensorboard<2.15,>=2.14 (from tensorflow-intel==2.14.0->tensorflow)
      Obtaining dependency information for tensorboard<2.15,>=2.14 from https://files.pythonhosted.org/packages/73/a2/66ed644f6ed1562e0285fcd959af17670ea313c8f331c46f79ee77187eb9/tensorboard-2.14.1-py3-none-any.whl.metadata
      Using cached tensorboard-2.14.1-py3-none-any.whl.metadata (1.7 kB)
    Collecting tensorflow-estimator<2.15,>=2.14.0 (from tensorflow-intel==2.14.0->tensorflow)
      Obtaining dependency information for tensorflow-estimator<2.15,>=2.14.0 from https://files.pythonhosted.org/packages/d1/da/4f264c196325bb6e37a6285caec5b12a03def489b57cc1fdac02bb6272cd/tensorflow_estimator-2.14.0-py2.py3-none-any.whl.metadata
      Using cached tensorflow_estimator-2.14.0-py2.py3-none-any.whl.metadata (1.3 kB)
    Collecting keras<2.15,>=2.14.0 (from tensorflow-intel==2.14.0->tensorflow)
      Obtaining dependency information for keras<2.15,>=2.14.0 from https://files.pythonhosted.org/packages/fe/58/34d4d8f1aa11120c2d36d7ad27d0526164b1a8ae45990a2fede31d0e59bf/keras-2.14.0-py3-none-any.whl.metadata
      Using cached keras-2.14.0-py3-none-any.whl.metadata (2.4 kB)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\users\kodal\anaconda3\lib\site-packages (from astunparse>=1.6.0->tensorflow-intel==2.14.0->tensorflow) (0.38.4)
    Collecting google-auth<3,>=1.6.3 (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow)
      Obtaining dependency information for google-auth<3,>=1.6.3 from https://files.pythonhosted.org/packages/39/7c/2e4fa55a99f83ef9ef229ac5d59c44ceb90e2d0145711590c0fa39669f32/google_auth-2.23.3-py2.py3-none-any.whl.metadata
      Using cached google_auth-2.23.3-py2.py3-none-any.whl.metadata (4.2 kB)
    Collecting google-auth-oauthlib<1.1,>=0.5 (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow)
      Using cached google_auth_oauthlib-1.0.0-py2.py3-none-any.whl (18 kB)
    Requirement already satisfied: markdown>=2.6.8 in c:\users\kodal\anaconda3\lib\site-packages (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (3.4.1)
    Requirement already satisfied: requests<3,>=2.21.0 in c:\users\kodal\anaconda3\lib\site-packages (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2.31.0)
    Collecting tensorboard-data-server<0.8.0,>=0.7.0 (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow)
      Obtaining dependency information for tensorboard-data-server<0.8.0,>=0.7.0 from https://files.pythonhosted.org/packages/7a/13/e503968fefabd4c6b2650af21e110aa8466fe21432cd7c43a84577a89438/tensorboard_data_server-0.7.2-py3-none-any.whl.metadata
      Using cached tensorboard_data_server-0.7.2-py3-none-any.whl.metadata (1.1 kB)
    Requirement already satisfied: werkzeug>=1.0.1 in c:\users\kodal\anaconda3\lib\site-packages (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (3.0.0)
    Collecting cachetools<6.0,>=2.0.0 (from google-auth<3,>=1.6.3->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow)
      Obtaining dependency information for cachetools<6.0,>=2.0.0 from https://files.pythonhosted.org/packages/a2/91/2d843adb9fbd911e0da45fbf6f18ca89d07a087c3daa23e955584f90ebf4/cachetools-5.3.2-py3-none-any.whl.metadata
      Using cached cachetools-5.3.2-py3-none-any.whl.metadata (5.2 kB)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\kodal\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (0.2.8)
    Collecting rsa<5,>=3.1.4 (from google-auth<3,>=1.6.3->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow)
      Using cached rsa-4.9-py3-none-any.whl (34 kB)
    Collecting requests-oauthlib>=0.7.0 (from google-auth-oauthlib<1.1,>=0.5->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow)
      Using cached requests_oauthlib-1.3.1-py2.py3-none-any.whl (23 kB)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\kodal\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\kodal\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\kodal\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (1.26.16)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\kodal\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2023.7.22)
    Requirement already satisfied: MarkupSafe>=2.1.1 in c:\users\kodal\anaconda3\lib\site-packages (from werkzeug>=1.0.1->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2.1.1)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in c:\users\kodal\anaconda3\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (0.4.8)
    Collecting oauthlib>=3.0.0 (from requests-oauthlib>=0.7.0->google-auth-oauthlib<1.1,>=0.5->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow)
      Using cached oauthlib-3.2.2-py3-none-any.whl (151 kB)
    Using cached tensorflow-2.14.0-cp311-cp311-win_amd64.whl (2.1 kB)
    Using cached tensorflow_intel-2.14.0-cp311-cp311-win_amd64.whl (284.2 MB)
    Using cached ml_dtypes-0.2.0-cp311-cp311-win_amd64.whl (938 kB)
    Using cached absl_py-2.0.0-py3-none-any.whl (130 kB)
    Using cached flatbuffers-23.5.26-py2.py3-none-any.whl (26 kB)
    Downloading grpcio-1.59.2-cp311-cp311-win_amd64.whl (3.7 MB)
       ---------------------------------------- 0.0/3.7 MB ? eta -:--:--
       ---------------------------------------- 0.0/3.7 MB 1.9 MB/s eta 0:00:02
       ---------------------------------------- 0.0/3.7 MB 1.9 MB/s eta 0:00:02
       - -------------------------------------- 0.1/3.7 MB 1.4 MB/s eta 0:00:03
       - -------------------------------------- 0.1/3.7 MB 1.4 MB/s eta 0:00:03
       --- ------------------------------------ 0.3/3.7 MB 1.5 MB/s eta 0:00:03
       ------ --------------------------------- 0.6/3.7 MB 2.4 MB/s eta 0:00:02
       ------------- -------------------------- 1.2/3.7 MB 4.2 MB/s eta 0:00:01
       ---------------- ----------------------- 1.5/3.7 MB 4.7 MB/s eta 0:00:01
       -------------------------- ------------- 2.4/3.7 MB 6.7 MB/s eta 0:00:01
       ----------------------------------- ---- 3.3/3.7 MB 8.0 MB/s eta 0:00:01
       -------------------------------------- - 3.5/3.7 MB 8.3 MB/s eta 0:00:01
       -------------------------------------- - 3.5/3.7 MB 8.3 MB/s eta 0:00:01
       ---------------------------------------- 3.7/3.7 MB 6.9 MB/s eta 0:00:00
    Using cached keras-2.14.0-py3-none-any.whl (1.7 MB)
    Using cached libclang-16.0.6-py2.py3-none-win_amd64.whl (24.4 MB)
    Using cached protobuf-4.24.4-cp310-abi3-win_amd64.whl (430 kB)
    Using cached tensorboard-2.14.1-py3-none-any.whl (5.5 MB)
    Using cached tensorflow_estimator-2.14.0-py2.py3-none-any.whl (440 kB)
    Using cached google_auth-2.23.3-py2.py3-none-any.whl (182 kB)
    Using cached tensorboard_data_server-0.7.2-py3-none-any.whl (2.4 kB)
    Using cached cachetools-5.3.2-py3-none-any.whl (9.3 kB)
    Installing collected packages: libclang, flatbuffers, termcolor, tensorflow-io-gcs-filesystem, tensorflow-estimator, tensorboard-data-server, rsa, protobuf, opt-einsum, oauthlib, ml-dtypes, keras, grpcio, google-pasta, gast, cachetools, astunparse, absl-py, requests-oauthlib, google-auth, google-auth-oauthlib, tensorboard, tensorflow-intel, tensorflow
    Successfully installed absl-py-2.0.0 astunparse-1.6.3 cachetools-5.3.2 flatbuffers-23.5.26 gast-0.5.4 google-auth-2.23.3 google-auth-oauthlib-1.0.0 google-pasta-0.2.0 grpcio-1.59.2 keras-2.14.0 libclang-16.0.6 ml-dtypes-0.2.0 oauthlib-3.2.2 opt-einsum-3.3.0 protobuf-4.24.4 requests-oauthlib-1.3.1 rsa-4.9 tensorboard-2.14.1 tensorboard-data-server-0.7.2 tensorflow-2.14.0 tensorflow-estimator-2.14.0 tensorflow-intel-2.14.0 tensorflow-io-gcs-filesystem-0.31.0 termcolor-2.3.0
    Note: you may need to restart the kernel to use updated packages.


### What is a Library?
Essentially a list of pre-written code that you can use to streamline and clean up your program.

Libraries can help simplify complex programs

APIS are specifications for how the procedures in a library behave, and how they can be used 

Documentations for an API/library is necessary in understanding the behaviors provided by the API/library and how to use them

Libraries that we will go over: Requests, Pillow, Pandas, Numpy, Scikit-Learn, TensorFlow, matplotlib.


### Required Installations
Please run the following commands in your vscode terminal in order to continue the lesson
- pip install numpy
- pip install matplotlib
- pip install scikit-learn
- pip install pillow
- pip install pandas
- pip install tensorflow
- pip install requests

### Images using requests and pillow libraries
'Requests' is focused on handling HTTP requests and web data while 'Pillow' is designed for data manipulation and analysis
It's common to see them used together in data-related assignments where data is fetched by HTTP requests using Requests and then processed and analyzed with Pandas.

Here's an example:


```python
# FOR THIS ONE I WAS UNABLE TO GET IT WORKING IN CLASS WHEN I TALKED TO ONE OF THE PEOPLE TEACHING IT
# THEY SAID IT WAS OKAY IF I DIDNT HAVE IT BECASUE WE COULDNT DEBUG, EVEN WHEN WE HAD PROPER IMAGE URL AND EVERYTHING SETUP


import requests
from PIL import Image
from io import BytesIO

# Step 1: Download an image using Requests
image_url = "https://example.com/path/to/your/image.jpg"  # Replace with the actual URL of the image you want to download
response = requests.get(image_url)

if response.status_code == 200:
    # Step 2: Process the downloaded image using Pillow
    image_data = BytesIO(response.content)  # Create an in-memory binary stream from the response content
    img = Image.open(image_data)  # Open the image using Pillow

    # Perform image processing tasks here, like resizing or applying filters
    img = img.resize((x, y))  # Resize the image and replace x,y with desired amounts

    # Step 3: Save the processed image using Pillow
    img.save("processed_image.jpg")  # Save the processed image to a file

    print("Image downloaded, processed, and saved.")
else:
    print(f"Failed to download image. Status code: {response.status_code}")

```

    Failed to download image. Status code: 404


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
#Code here
import requests
from PIL import Image
from io import BytesIO

# Step 1: Download an image using Requests
image_url = "https://cdn.vox-cdn.com/thumbor/hhnSE8SDm8Bg92fG_gb3J9gTUjU=/0x0:6000x4000/1200x800/filters:focal(2958x758:3918x1718)/cdn.vox-cdn.com/uploads/chorus_image/image/72365465/1258245670.0.jpg"  # Replace with the actual URL of the image you want to download
response = requests.get(image_url)

if response.status_code == 200:
    # Step 2: Process the downloaded image using Pillow
    image_data = BytesIO(response.content)  # Create an in-memory binary stream from the response content
    img = Image.open(image_data)  # Open the image using Pillow

    # Perform image processing tasks here, like resizing or applying filters
    img = img.resize((670,490))  # Resize the image and replace x,y with desired amounts

    # Step 3: Save the processed image using Pillow
    img.save("processed_image.jpg")  # Save the processed image to a file

    print("Image downloaded, processed, and saved.")
else:
    print(f"Failed to download image. Status code: {response.status_code}")

```

    Image downloaded, processed, and saved.


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
    


### Hack 2
Using the data from the numpy library, create a visual graph using different matplotlib functions.


```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data for two lines
x = np.linspace(0, 10, 50)  # Create an array of values from 0 to 10
y1 = 2 * x + 1  # Set of data poits

# Create and display a plot using Matplotlib
# Create a simple line plot using Matplotlib
plt.plot(x, y1, label='Line', color='blue', linestyle='-')  # Create the plot
plt.title('Line')  # Set the title
plt.xlabel('x')  # Label for the x-axis
plt.ylabel('y')  # Label for the y-axis
plt.grid(True)  # Display a grid
plt.legend()  # Show the legend
plt.show()  # Display the plot
# your code here

```


    
![png](output_13_0.png)
    


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

    Epoch 1/100
    3/3 - 0s - loss: 3.7797 - 293ms/epoch - 98ms/step
    Epoch 2/100
    3/3 - 0s - loss: 3.7674 - 5ms/epoch - 2ms/step
    Epoch 3/100
    3/3 - 0s - loss: 3.7556 - 2ms/epoch - 770us/step
    Epoch 4/100
    3/3 - 0s - loss: 3.7436 - 9ms/epoch - 3ms/step
    Epoch 5/100
    3/3 - 0s - loss: 3.7314 - 2ms/epoch - 562us/step
    Epoch 6/100
    3/3 - 0s - loss: 3.7197 - 5ms/epoch - 2ms/step
    Epoch 7/100
    3/3 - 0s - loss: 3.7079 - 8ms/epoch - 3ms/step
    Epoch 8/100
    3/3 - 0s - loss: 3.6963 - 24ms/epoch - 8ms/step
    Epoch 9/100
    3/3 - 0s - loss: 3.6844 - 8ms/epoch - 3ms/step
    Epoch 10/100
    3/3 - 0s - loss: 3.6727 - 3ms/epoch - 849us/step
    Epoch 11/100
    3/3 - 0s - loss: 3.6609 - 5ms/epoch - 2ms/step
    Epoch 12/100
    3/3 - 0s - loss: 3.6490 - 7ms/epoch - 2ms/step
    Epoch 13/100
    3/3 - 0s - loss: 3.6375 - 2ms/epoch - 787us/step
    Epoch 14/100
    3/3 - 0s - loss: 3.6256 - 8ms/epoch - 3ms/step
    Epoch 15/100
    3/3 - 0s - loss: 3.6140 - 3ms/epoch - 1ms/step
    Epoch 16/100
    3/3 - 0s - loss: 3.6023 - 7ms/epoch - 2ms/step
    Epoch 17/100
    3/3 - 0s - loss: 3.5910 - 7ms/epoch - 2ms/step
    Epoch 18/100
    3/3 - 0s - loss: 3.5793 - 2ms/epoch - 828us/step
    Epoch 19/100
    3/3 - 0s - loss: 3.5679 - 9ms/epoch - 3ms/step
    Epoch 20/100
    3/3 - 0s - loss: 3.5562 - 3ms/epoch - 1ms/step
    Epoch 21/100
    3/3 - 0s - loss: 3.5447 - 4ms/epoch - 1ms/step
    Epoch 22/100
    3/3 - 0s - loss: 3.5334 - 4ms/epoch - 1ms/step
    Epoch 23/100
    3/3 - 0s - loss: 3.5224 - 1ms/epoch - 352us/step
    Epoch 24/100
    3/3 - 0s - loss: 3.5107 - 3ms/epoch - 1ms/step
    Epoch 25/100
    3/3 - 0s - loss: 3.4993 - 4ms/epoch - 1ms/step
    Epoch 26/100
    3/3 - 0s - loss: 3.4881 - 2ms/epoch - 653us/step
    Epoch 27/100
    3/3 - 0s - loss: 3.4769 - 8ms/epoch - 3ms/step
    Epoch 28/100
    3/3 - 0s - loss: 3.4657 - 3ms/epoch - 1ms/step
    Epoch 29/100
    3/3 - 0s - loss: 3.4543 - 8ms/epoch - 3ms/step
    Epoch 30/100
    3/3 - 0s - loss: 3.4435 - 4ms/epoch - 1ms/step
    Epoch 31/100
    3/3 - 0s - loss: 3.4321 - 6ms/epoch - 2ms/step
    Epoch 32/100
    3/3 - 0s - loss: 3.4211 - 6ms/epoch - 2ms/step
    Epoch 33/100
    3/3 - 0s - loss: 3.4099 - 6ms/epoch - 2ms/step
    Epoch 34/100
    3/3 - 0s - loss: 3.3993 - 6ms/epoch - 2ms/step
    Epoch 35/100
    3/3 - 0s - loss: 3.3879 - 0s/epoch - 0s/step
    Epoch 36/100
    3/3 - 0s - loss: 3.3770 - 4ms/epoch - 1ms/step
    Epoch 37/100
    3/3 - 0s - loss: 3.3663 - 0s/epoch - 0s/step
    Epoch 38/100
    3/3 - 0s - loss: 3.3551 - 3ms/epoch - 1ms/step
    Epoch 39/100
    3/3 - 0s - loss: 3.3444 - 928us/epoch - 309us/step
    Epoch 40/100
    3/3 - 0s - loss: 3.3335 - 2ms/epoch - 754us/step
    Epoch 41/100
    3/3 - 0s - loss: 3.3226 - 14ms/epoch - 5ms/step
    Epoch 42/100
    3/3 - 0s - loss: 3.3120 - 10ms/epoch - 3ms/step
    Epoch 43/100
    3/3 - 0s - loss: 3.3012 - 10ms/epoch - 3ms/step
    Epoch 44/100
    3/3 - 0s - loss: 3.2906 - 14ms/epoch - 5ms/step
    Epoch 45/100
    3/3 - 0s - loss: 3.2797 - 9ms/epoch - 3ms/step
    Epoch 46/100
    3/3 - 0s - loss: 3.2691 - 6ms/epoch - 2ms/step
    Epoch 47/100
    3/3 - 0s - loss: 3.2583 - 3ms/epoch - 868us/step
    Epoch 48/100
    3/3 - 0s - loss: 3.2477 - 8ms/epoch - 3ms/step
    Epoch 49/100
    3/3 - 0s - loss: 3.2371 - 3ms/epoch - 975us/step
    Epoch 50/100
    3/3 - 0s - loss: 3.2266 - 9ms/epoch - 3ms/step
    Epoch 51/100
    3/3 - 0s - loss: 3.2159 - 3ms/epoch - 1ms/step
    Epoch 52/100
    3/3 - 0s - loss: 3.2055 - 7ms/epoch - 2ms/step
    Epoch 53/100
    3/3 - 0s - loss: 3.1952 - 5ms/epoch - 2ms/step
    Epoch 54/100
    3/3 - 0s - loss: 3.1845 - 6ms/epoch - 2ms/step
    Epoch 55/100
    3/3 - 0s - loss: 3.1743 - 8ms/epoch - 3ms/step
    Epoch 56/100
    3/3 - 0s - loss: 3.1638 - 3ms/epoch - 1ms/step
    Epoch 57/100
    3/3 - 0s - loss: 3.1534 - 6ms/epoch - 2ms/step
    Epoch 58/100
    3/3 - 0s - loss: 3.1432 - 4ms/epoch - 1ms/step
    Epoch 59/100
    3/3 - 0s - loss: 3.1329 - 6ms/epoch - 2ms/step
    Epoch 60/100
    3/3 - 0s - loss: 3.1227 - 6ms/epoch - 2ms/step
    Epoch 61/100
    3/3 - 0s - loss: 3.1122 - 5ms/epoch - 2ms/step
    Epoch 62/100
    3/3 - 0s - loss: 3.1021 - 6ms/epoch - 2ms/step
    Epoch 63/100
    3/3 - 0s - loss: 3.0919 - 5ms/epoch - 2ms/step
    Epoch 64/100
    3/3 - 0s - loss: 3.0816 - 6ms/epoch - 2ms/step
    Epoch 65/100
    3/3 - 0s - loss: 3.0716 - 0s/epoch - 0s/step
    Epoch 66/100
    3/3 - 0s - loss: 3.0615 - 5ms/epoch - 2ms/step
    Epoch 67/100
    3/3 - 0s - loss: 3.0513 - 5ms/epoch - 2ms/step
    Epoch 68/100
    3/3 - 0s - loss: 3.0413 - 4ms/epoch - 1ms/step
    Epoch 69/100
    3/3 - 0s - loss: 3.0315 - 5ms/epoch - 2ms/step
    Epoch 70/100
    3/3 - 0s - loss: 3.0214 - 0s/epoch - 0s/step
    Epoch 71/100
    3/3 - 0s - loss: 3.0115 - 2ms/epoch - 678us/step
    Epoch 72/100
    3/3 - 0s - loss: 3.0015 - 1ms/epoch - 406us/step
    Epoch 73/100
    3/3 - 0s - loss: 2.9916 - 2ms/epoch - 740us/step
    Epoch 74/100
    3/3 - 0s - loss: 2.9817 - 5ms/epoch - 2ms/step
    Epoch 75/100
    3/3 - 0s - loss: 2.9718 - 9ms/epoch - 3ms/step
    Epoch 76/100
    3/3 - 0s - loss: 2.9620 - 5ms/epoch - 2ms/step
    Epoch 77/100
    3/3 - 0s - loss: 2.9521 - 9ms/epoch - 3ms/step
    Epoch 78/100
    3/3 - 0s - loss: 2.9423 - 8ms/epoch - 3ms/step
    Epoch 79/100
    3/3 - 0s - loss: 2.9325 - 1ms/epoch - 341us/step
    Epoch 80/100
    3/3 - 0s - loss: 2.9227 - 3ms/epoch - 1ms/step
    Epoch 81/100
    3/3 - 0s - loss: 2.9131 - 5ms/epoch - 2ms/step
    Epoch 82/100
    3/3 - 0s - loss: 2.9032 - 3ms/epoch - 899us/step
    Epoch 83/100
    3/3 - 0s - loss: 2.8935 - 7ms/epoch - 2ms/step
    Epoch 84/100
    3/3 - 0s - loss: 2.8837 - 4ms/epoch - 1ms/step
    Epoch 85/100
    3/3 - 0s - loss: 2.8742 - 3ms/epoch - 1ms/step
    Epoch 86/100
    3/3 - 0s - loss: 2.8645 - 8ms/epoch - 3ms/step
    Epoch 87/100
    3/3 - 0s - loss: 2.8549 - 4ms/epoch - 1ms/step
    Epoch 88/100
    3/3 - 0s - loss: 2.8453 - 8ms/epoch - 3ms/step
    Epoch 89/100
    3/3 - 0s - loss: 2.8357 - 8ms/epoch - 3ms/step
    Epoch 90/100
    3/3 - 0s - loss: 2.8265 - 4ms/epoch - 1ms/step
    Epoch 91/100
    3/3 - 0s - loss: 2.8166 - 7ms/epoch - 2ms/step
    Epoch 92/100
    3/3 - 0s - loss: 2.8072 - 0s/epoch - 0s/step
    Epoch 93/100
    3/3 - 0s - loss: 2.7977 - 1ms/epoch - 482us/step
    Epoch 94/100
    3/3 - 0s - loss: 2.7884 - 6ms/epoch - 2ms/step
    Epoch 95/100
    3/3 - 0s - loss: 2.7788 - 7ms/epoch - 2ms/step
    Epoch 96/100
    3/3 - 0s - loss: 2.7696 - 5ms/epoch - 2ms/step
    Epoch 97/100
    3/3 - 0s - loss: 2.7601 - 8ms/epoch - 3ms/step
    Epoch 98/100
    3/3 - 0s - loss: 2.7507 - 8ms/epoch - 3ms/step
    Epoch 99/100
    3/3 - 0s - loss: 2.7415 - 10ms/epoch - 3ms/step
    Epoch 100/100
    3/3 - 0s - loss: 2.7322 - 0s/epoch - 0s/step
    1/1 [==============================] - 0s 66ms/step
    Mean Squared Error: 2.7604


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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a simple linear regression model using TensorFlow and Keras
model = keras.Sequential([
    layers.Input(shape=(2,)),  # Input shape matches the number of features (2)
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

    Epoch 1/100
    3/3 - 0s - loss: 171448303616.0000 - 451ms/epoch - 150ms/step
    Epoch 2/100
    3/3 - 0s - loss: 171448303616.0000 - 9ms/epoch - 3ms/step
    Epoch 3/100
    3/3 - 0s - loss: 171448303616.0000 - 9ms/epoch - 3ms/step
    Epoch 4/100
    3/3 - 0s - loss: 171448303616.0000 - 10ms/epoch - 3ms/step
    Epoch 5/100
    3/3 - 0s - loss: 171448303616.0000 - 8ms/epoch - 3ms/step
    Epoch 6/100
    3/3 - 0s - loss: 171448303616.0000 - 11ms/epoch - 4ms/step
    Epoch 7/100
    3/3 - 0s - loss: 171448303616.0000 - 9ms/epoch - 3ms/step
    Epoch 8/100
    3/3 - 0s - loss: 171448303616.0000 - 11ms/epoch - 4ms/step
    Epoch 9/100
    3/3 - 0s - loss: 171448303616.0000 - 9ms/epoch - 3ms/step
    Epoch 10/100
    3/3 - 0s - loss: 171448303616.0000 - 8ms/epoch - 3ms/step
    Epoch 11/100
    3/3 - 0s - loss: 171448287232.0000 - 9ms/epoch - 3ms/step
    Epoch 12/100
    3/3 - 0s - loss: 171448287232.0000 - 11ms/epoch - 4ms/step
    Epoch 13/100
    3/3 - 0s - loss: 171448287232.0000 - 13ms/epoch - 4ms/step
    Epoch 14/100
    3/3 - 0s - loss: 171448287232.0000 - 8ms/epoch - 3ms/step
    Epoch 15/100
    3/3 - 0s - loss: 171448270848.0000 - 8ms/epoch - 3ms/step
    Epoch 16/100
    3/3 - 0s - loss: 171448270848.0000 - 9ms/epoch - 3ms/step
    Epoch 17/100
    3/3 - 0s - loss: 171448287232.0000 - 7ms/epoch - 2ms/step
    Epoch 18/100
    3/3 - 0s - loss: 171448270848.0000 - 7ms/epoch - 2ms/step
    Epoch 19/100
    3/3 - 0s - loss: 171448270848.0000 - 6ms/epoch - 2ms/step
    Epoch 20/100
    3/3 - 0s - loss: 171448270848.0000 - 8ms/epoch - 3ms/step
    Epoch 21/100
    3/3 - 0s - loss: 171448254464.0000 - 6ms/epoch - 2ms/step
    Epoch 22/100
    3/3 - 0s - loss: 171448254464.0000 - 6ms/epoch - 2ms/step
    Epoch 23/100
    3/3 - 0s - loss: 171448254464.0000 - 3ms/epoch - 1ms/step
    Epoch 24/100
    3/3 - 0s - loss: 171448254464.0000 - 1ms/epoch - 341us/step
    Epoch 25/100
    3/3 - 0s - loss: 171448254464.0000 - 10ms/epoch - 3ms/step
    Epoch 26/100
    3/3 - 0s - loss: 171448254464.0000 - 11ms/epoch - 4ms/step
    Epoch 27/100
    3/3 - 0s - loss: 171448254464.0000 - 10ms/epoch - 3ms/step
    Epoch 28/100
    3/3 - 0s - loss: 171448238080.0000 - 9ms/epoch - 3ms/step
    Epoch 29/100
    3/3 - 0s - loss: 171448238080.0000 - 8ms/epoch - 3ms/step
    Epoch 30/100
    3/3 - 0s - loss: 171448238080.0000 - 8ms/epoch - 3ms/step
    Epoch 31/100
    3/3 - 0s - loss: 171448238080.0000 - 9ms/epoch - 3ms/step
    Epoch 32/100
    3/3 - 0s - loss: 171448238080.0000 - 8ms/epoch - 3ms/step
    Epoch 33/100
    3/3 - 0s - loss: 171448238080.0000 - 21ms/epoch - 7ms/step
    Epoch 34/100
    3/3 - 0s - loss: 171448238080.0000 - 25ms/epoch - 8ms/step
    Epoch 35/100
    3/3 - 0s - loss: 171448238080.0000 - 14ms/epoch - 5ms/step
    Epoch 36/100
    3/3 - 0s - loss: 171448238080.0000 - 10ms/epoch - 3ms/step
    Epoch 37/100
    3/3 - 0s - loss: 171448238080.0000 - 15ms/epoch - 5ms/step
    Epoch 38/100
    3/3 - 0s - loss: 171448221696.0000 - 5ms/epoch - 2ms/step
    Epoch 39/100
    3/3 - 0s - loss: 171448221696.0000 - 11ms/epoch - 4ms/step
    Epoch 40/100
    3/3 - 0s - loss: 171448205312.0000 - 2ms/epoch - 703us/step
    Epoch 41/100
    3/3 - 0s - loss: 171448221696.0000 - 14ms/epoch - 5ms/step
    Epoch 42/100
    3/3 - 0s - loss: 171448205312.0000 - 8ms/epoch - 3ms/step
    Epoch 43/100
    3/3 - 0s - loss: 171448205312.0000 - 7ms/epoch - 2ms/step
    Epoch 44/100
    3/3 - 0s - loss: 171448205312.0000 - 6ms/epoch - 2ms/step
    Epoch 45/100
    3/3 - 0s - loss: 171448205312.0000 - 7ms/epoch - 2ms/step
    Epoch 46/100
    3/3 - 0s - loss: 171448188928.0000 - 6ms/epoch - 2ms/step
    Epoch 47/100
    3/3 - 0s - loss: 171448188928.0000 - 15ms/epoch - 5ms/step
    Epoch 48/100
    3/3 - 0s - loss: 171448205312.0000 - 11ms/epoch - 4ms/step
    Epoch 49/100
    3/3 - 0s - loss: 171448172544.0000 - 9ms/epoch - 3ms/step
    Epoch 50/100
    3/3 - 0s - loss: 171448172544.0000 - 8ms/epoch - 3ms/step
    Epoch 51/100
    3/3 - 0s - loss: 171448172544.0000 - 8ms/epoch - 3ms/step
    Epoch 52/100
    3/3 - 0s - loss: 171448172544.0000 - 7ms/epoch - 2ms/step
    Epoch 53/100
    3/3 - 0s - loss: 171448172544.0000 - 9ms/epoch - 3ms/step
    Epoch 54/100
    3/3 - 0s - loss: 171448172544.0000 - 11ms/epoch - 4ms/step
    Epoch 55/100
    3/3 - 0s - loss: 171448172544.0000 - 7ms/epoch - 2ms/step
    Epoch 56/100
    3/3 - 0s - loss: 171448172544.0000 - 9ms/epoch - 3ms/step
    Epoch 57/100
    3/3 - 0s - loss: 171448172544.0000 - 10ms/epoch - 3ms/step
    Epoch 58/100
    3/3 - 0s - loss: 171448156160.0000 - 11ms/epoch - 4ms/step
    Epoch 59/100
    3/3 - 0s - loss: 171448172544.0000 - 8ms/epoch - 3ms/step
    Epoch 60/100
    3/3 - 0s - loss: 171448156160.0000 - 7ms/epoch - 2ms/step
    Epoch 61/100
    3/3 - 0s - loss: 171448156160.0000 - 8ms/epoch - 3ms/step
    Epoch 62/100
    3/3 - 0s - loss: 171448172544.0000 - 11ms/epoch - 4ms/step
    Epoch 63/100
    3/3 - 0s - loss: 171448156160.0000 - 14ms/epoch - 5ms/step
    Epoch 64/100
    3/3 - 0s - loss: 171448156160.0000 - 5ms/epoch - 2ms/step
    Epoch 65/100
    3/3 - 0s - loss: 171448156160.0000 - 11ms/epoch - 4ms/step
    Epoch 66/100
    3/3 - 0s - loss: 171448123392.0000 - 9ms/epoch - 3ms/step
    Epoch 67/100
    3/3 - 0s - loss: 171448123392.0000 - 11ms/epoch - 4ms/step
    Epoch 68/100
    3/3 - 0s - loss: 171448123392.0000 - 9ms/epoch - 3ms/step
    Epoch 69/100
    3/3 - 0s - loss: 171448139776.0000 - 8ms/epoch - 3ms/step
    Epoch 70/100
    3/3 - 0s - loss: 171448123392.0000 - 9ms/epoch - 3ms/step
    Epoch 71/100
    3/3 - 0s - loss: 171448123392.0000 - 11ms/epoch - 4ms/step
    Epoch 72/100
    3/3 - 0s - loss: 171448123392.0000 - 11ms/epoch - 4ms/step
    Epoch 73/100
    3/3 - 0s - loss: 171448123392.0000 - 6ms/epoch - 2ms/step
    Epoch 74/100
    3/3 - 0s - loss: 171448123392.0000 - 9ms/epoch - 3ms/step
    Epoch 75/100
    3/3 - 0s - loss: 171448123392.0000 - 8ms/epoch - 3ms/step
    Epoch 76/100
    3/3 - 0s - loss: 171448123392.0000 - 11ms/epoch - 4ms/step
    Epoch 77/100
    3/3 - 0s - loss: 171448107008.0000 - 6ms/epoch - 2ms/step
    Epoch 78/100
    3/3 - 0s - loss: 171448107008.0000 - 8ms/epoch - 3ms/step
    Epoch 79/100
    3/3 - 0s - loss: 171448107008.0000 - 3ms/epoch - 1ms/step
    Epoch 80/100
    3/3 - 0s - loss: 171448107008.0000 - 8ms/epoch - 3ms/step
    Epoch 81/100
    3/3 - 0s - loss: 171448107008.0000 - 12ms/epoch - 4ms/step
    Epoch 82/100
    3/3 - 0s - loss: 171448107008.0000 - 16ms/epoch - 5ms/step
    Epoch 83/100
    3/3 - 0s - loss: 171448090624.0000 - 20ms/epoch - 7ms/step
    Epoch 84/100
    3/3 - 0s - loss: 171448090624.0000 - 20ms/epoch - 7ms/step
    Epoch 85/100
    3/3 - 0s - loss: 171448090624.0000 - 16ms/epoch - 5ms/step
    Epoch 86/100
    3/3 - 0s - loss: 171448090624.0000 - 11ms/epoch - 4ms/step
    Epoch 87/100
    3/3 - 0s - loss: 171448090624.0000 - 8ms/epoch - 3ms/step
    Epoch 88/100
    3/3 - 0s - loss: 171448090624.0000 - 8ms/epoch - 3ms/step
    Epoch 89/100
    3/3 - 0s - loss: 171448090624.0000 - 6ms/epoch - 2ms/step
    Epoch 90/100
    3/3 - 0s - loss: 171448074240.0000 - 9ms/epoch - 3ms/step
    Epoch 91/100
    3/3 - 0s - loss: 171448074240.0000 - 5ms/epoch - 2ms/step
    Epoch 92/100
    3/3 - 0s - loss: 171448057856.0000 - 9ms/epoch - 3ms/step
    Epoch 93/100
    3/3 - 0s - loss: 171448074240.0000 - 10ms/epoch - 3ms/step
    Epoch 94/100
    3/3 - 0s - loss: 171448074240.0000 - 9ms/epoch - 3ms/step
    Epoch 95/100
    3/3 - 0s - loss: 171448057856.0000 - 9ms/epoch - 3ms/step
    Epoch 96/100
    3/3 - 0s - loss: 171448057856.0000 - 8ms/epoch - 3ms/step
    Epoch 97/100
    3/3 - 0s - loss: 171448041472.0000 - 9ms/epoch - 3ms/step
    Epoch 98/100
    3/3 - 0s - loss: 171448057856.0000 - 6ms/epoch - 2ms/step
    Epoch 99/100
    3/3 - 0s - loss: 171448057856.0000 - 4ms/epoch - 1ms/step
    Epoch 100/100
    3/3 - 0s - loss: 171448041472.0000 - 3ms/epoch - 961us/step
    1/1 [==============================] - 0s 55ms/step
    Mean Squared Error: 158481932535.6291


## HOMEWORK 1

Create a GPA calculator using Pandas and Matplot libraries and make:
1) A dataframe
2) A specified dictionary
3) and a print function that outputs the final GPA

Extra points can be earned with creativity.


```python
import pandas as pd
import matplotlib.pyplot as plt

# Define a dictionary with grade values
grade_dict = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}

data = {'Subject': ['AP Calc', 'APCSP', 'APUSH', 'Photo 1', 'APEL'],
        'Grade': ['A', 'B', 'A', 'A', 'A'],
        'Teacher': ['Bhueler', 'Mortenson', 'Swanson', 'McClusky', 'Dafoe']}

# Function to calculate GPA
def calculate_gpa(data):
    FinalGPA = 0
    for index, row in data.iterrows():
        class_name = row['Subject']
        letter_grade = str(row['Grade'])
        GPA = grade_dict.get(letter_grade, 0)  # Use get() to handle missing grades
        FinalGPA += GPA
    avgGPA = FinalGPA / len(data)
    print(f"Average GPA is {avgGPA}")
    return avgGPA

df = pd.DataFrame(data)

# Calculate GPA
gpa = calculate_gpa(df)

# Display the DataFrame
print(df)

# Plotting the GPA for each class
fig, ax = plt.subplots()
bar_width = 0.4

class_names = df['Subject']
gpa_values = [grade_dict.get(grade, 0) for grade in df['Grade']]

bars = ax.bar(class_names, gpa_values, color='blue')

# Add labels and title
ax.set_ylabel('GPA')
ax.set_xlabel('Classes')
ax.set_title('GPA for Each Class')

# Add data values on top of the bars
for bar, gpa_value in zip(bars, gpa_values):
    height = bar.get_height()
    ax.annotate('{}'.format(gpa_value),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

plt.show()

```

    Average GPA is 3.8
       Subject Grade    Teacher
    0  AP Calc     A    Bhueler
    1    APCSP     B  Mortenson
    2    APUSH     A    Swanson
    3  Photo 1     A   McClusky
    4     APEL     A      Dafoe



    
![png](output_21_1.png)
    


## HOMEWORK 2

Import and use the "random" library to generate 50 different points from the range 0-100, then display the randomized data using a scatter plot.

Extra points can be earned with creativity.


```python
import random
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 100, 50)

setofvalues = []
count = 0
while count < 50:
    count += 1
    setofvalues.append(random.randint(0, 100))
y2 = setofvalues

plt.scatter(x, y2, color='green', s=18, alpha=0.5)
plt.title('Data')
plt.xlabel('')
plt.ylabel('')
plt.grid(True)
plt.show()
```


    
![png](output_23_0.png)
    


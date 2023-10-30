---
toc: True
comments: True
layout: post
title: Base64 blog
type: hacks
courses: {'csp': {'week': 9}}
---

```python
import base64

# Open the image file in binary mode
with open("image.jpeg", "rb") as image_file:
    # Read the binary data of the image file
    binary_data = image_file.read()
    # Encode the binary data into Base64
    base64_encoded = base64.b64encode(binary_data).decode("utf-8")

# Now, base64_encoded contains the Base64 representation of the image
print(base64_encoded)
#MY OPTIMAL SOLUTION
```

# **Navigating the Challenges of Base64 Encoding and POST Requests: My Practical Journey**

In my exploration of web development, the integration of images into applications demanded a deep understanding of techniques like Base64 encoding and precise handling of POST requests. As a high school coder, I encountered both challenges and enlightenment during this journey.

## **Encountering Base64 Encoding**

Base64 encoding, a fundamental mechanism for representing binary data in ASCII text format, fascinated me. Its role in embedding images directly into web applications is crucial. However, my journey took an unexpected turn when a corrupted image disrupted the encoding process, hindering the functionality of my project.

## **Exploring Base64 Encoding Methods**

During this process, I explored multiple methods for Base64 encoding, each with its own advantages:

### **1. Online Generator through Cloud:**

One approach was using online tools and cloud services. These platforms offer user-friendly interfaces where you can upload an image, and they generate the Base64 encoded string for you. While convenient, this method might not be suitable for all situations, especially when dealing with sensitive data or when automation is required.

### **2. Manual Encoding via Command Prompt:**

For a more hands-on approach, I experimented with manual encoding using the command prompt or terminal. This method involved using built-in commands to read the binary data of an image file and then encode it into Base64. While providing more control, it proved to be cumbersome and time-consuming, especially for larger projects or multiple images.

### **3. Python Library: The Elegant Solution**

Amidst the challenges, I discovered an elegant solution through Python. As mentioned earlier, we had initial issues with JPG files, which led to image corruption during the encoding process. By converting the images to the JPEG format, we resolved this problem. Moreover, using Python's built-in `base64` library, I could effortlessly encode images into Base64 with minimal code, exemplifying the elegance and power of Python in handling complex tasks with simplicity.

### **4. JPG solution**

We had an issue where all my files were becoming corrupted the second I put them into vscode so then we converted them from JPEG to JPG which was able to fix it because apparently vscode has issues.

### **5. Links to my code**
**Commit 1:**
[Commit 5b9a852d8bea952155e6da7489079d8fe12b1e35](https://github.com/will-w-cheng/team-influencer-innovator-backend/commit/5b9a852d8bea952155e6da7489079d8fe12b1e35)

**Commit 2 (Full Application):**
[Commit bc8e899d55c9c6aca4f8cd313add2ce64cea5bc0](https://github.com/will-w-cheng/team-influencer-innovator-backend/commit/bc8e899d55c9c6aca4f8cd313add2ce64cea5bc0)

**GitHub Issue:**
[Issue #1](https://github.com/will-w-cheng/team-influencer-innovator-backend/issues/1)




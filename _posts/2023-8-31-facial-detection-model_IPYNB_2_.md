---
comments: True
layout: post
title: Python AI model with mediapipe and opencv
description: Using your default camera and mediapipe. The code is able to access your camera and spatially analyze your face so that, we can see how you look before you take the photo and if your face is perfectly facing the camera or not
type: hacks
courses: {'csp': {'week': 2}}
---

```python
import cv2
import mediapipe as mp

# Initialize the FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check for key press
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        # Press 'q' to exit the script
        break
    elif key & 0xFF == ord(' '):
        # Press 'SPACE' to take a photo and save the original frame as "opencv_frame.png"
        cv2.imwrite("opencv_frame.png", frame.copy())  # Save the original frame
        print("Photo taken and saved as 'opencv_frame.png'")

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with FaceMesh
    results = face_mesh.process(frame_rgb)

    # Check if any face was detected
    if results.multi_face_landmarks:
        # Loop over each detected face
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                # Draw circles at each facial landmark point
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    # Display the frame with FaceMesh processing for the live camera feed
    cv2.imshow('FaceMesh Detection (Live Feed)', frame)

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

```

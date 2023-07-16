# ObjectDetectionCV2
We attempted to perform object detection using the OpenCV library with the help of a webcam.

# WHAT'S THE OPENCV?

OpenCV (Open Source Computer Vision Library) is a popular open-source library for computer vision and image processing. It provides a range of functions and tools for developing computer vision applications in C++, Python, Java, and other programming languages.
OpenCV enables tasks such as image processing, video analysis, object detection, face recognition, image correction, camera calibration, stereo imaging, artificial intelligence algorithms, and many other image processing and computer vision tasks. It supports various image formats and can utilize different hardware resources, such as CPU or GPU acceleration.
With OpenCV, users can capture, process, analyze, and visualize images and videos. It offers user-friendly interfaces, powerful functionalities, and a large community support, making it a widely adopted solution for image processing.

![VNtaG](https://github.com/ahmetdzdrr/ObjectDetectionCV2/assets/117534684/dddf84d3-fda7-4aba-8e09-49d99dbc60f7)

# CODE SOURCE

The code initializes the webcam capture using cv2.VideoCapture() and sets the desired resolution.
It loads the class names from a file (coco.names in this case) that contains the names of the objects to be detected.
The pre-trained model is loaded using cv2.dnn_DetectionModel() with the corresponding weight and configuration files.
The input size, scale, mean, and channel swapping are set for the model.
Inside the infinite loop, it reads frames from the webcam using cap.read().
The model.detect() function is called to perform object detection on the captured frame, providing the confidence threshold.
Detected objects are processed by iterating over the results and drawing bounding boxes and labels on the frame.
The annotated frame is displayed using cv2.imshow().
If the 'q' key is pressed, the loop is terminated and the program exits.
Finally, the webcam capture is released and the windows are closed using cap.release() and cv2.destroyAllWindows().

# RESULT

![Ekran görüntüsü 2023-07-16 185544](https://github.com/ahmetdzdrr/ObjectDetectionCV2/assets/117534684/21039dc9-6974-483f-b468-1813151b32fd)



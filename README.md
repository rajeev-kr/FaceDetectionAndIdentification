# Face Detection and Identification
Face Detection and Identification using [KNN Classifier Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) and [Haarcascade Frontal Face Model](https://github.com/opencv/opencv/tree/master/data/haarcascades).

## Requirements :
- Python, NumPy, OpenCV2

## Use :
### Data Generation
```python
python Data_Creation_from_HCC.py
```
    Enter the name of the person in the frame.
    Let it capture approx 50 snips in good lighting condition.
    You can see the snip count in the terminal.
    Press q to exit capture mode.
![](./img/data_creation.png)
![](./img/enter_name.png)
![](./img/face_capture.png)
### Face Recognition & Identification
```python
python Face_Detection_Using_KNN_Classifier_and_HCC_FFModel.py
```
    Put the people in the frame, the system will recognize their faces and names.
![](./img/detect.png)
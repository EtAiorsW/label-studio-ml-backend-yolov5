## YOLOv5 backend for Label Studio with TensorRT support
### requirements
```
label-studio = 1.8.2/1.10.0rc6
label-studio-ml = 1.0.9
TensorRT >= 8.6.1
pytorch >= 2.1.1
opencv-python ~= 4.8.1.78
```
!!!There are some bugs in label-studio = 1.9.x version!!!

### model.py
```
model_name : model engine file path
class_path : yaml file path from yolov5
```
#### for example 
NG.yaml:
```
# Classes
names:
  0: NG
```
### runs
```
label-studio-ml start label_studio_ml_backend_yolov5
```
### reference
```
https://github.com/ultralytics/yolov5
https://github.com/HumanSignal/label-studio
https://github.com/HumanSignal/label-studio-ml-backend
```

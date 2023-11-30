## YOLOv5 backend for Label Studio with TensorRT support
### requirements
```
label-studio = 1.8.2 \
label-studio-ml = 1.0.9
```
There are some bugs in higher version

### model.py
#### model_name : model engine file path
#### class_path : yaml file path from yolov5
#### for example
```
# Classes
names:
  0: NG
```
### runs
```
label-studio-ml start label_studio_ml_backend_yolov5
```

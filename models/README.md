## YOLOv5 TF Light

To get yolov5 tflite models, you can use [`yolov5/export.py`](https://github.com/ultralytics/yolov5/blob/master/export.py).

For example;

```bash
git clone git@github.com:ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt

python export.py --weights yolov5n.pt --include  tflite
```

For mode details, see the [official release note](https://github.com/ultralytics/yolov5/releases).
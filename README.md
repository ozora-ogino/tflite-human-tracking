<h1 align="center">
  TensorFlow Light Human Tracking
</h1>

<div align="center">
  <img src="./outputs/example_yolov5l.gif" width="60%">
</div>

The motivation of TensorFlow Lite Human Tracking is developing person tacking system for edge camera.
For example, count the number of visitors.

To track and detect people over frames, SORT is adopted.

Currently [YOLOv5](https://github.com/ultralytics/yolov5) models are supported for object detection.
To get YOLOv5 tflite model, see [`models/README.md`](./models/README.md)

## <div align="center">Quick Start Example</div>

```bash
git clone git@github.com:ozora-ogino/tflite-human-tracking.git
cd tflite-human-tracking
python main.py --src ./data/<YOUR_VIDEO_FILE>.mp4 --model ./models/<YOLOV5_MODEL>.tflite

# Set directions.
# For the value of direction you can choose one of 'bottom', 'top', right', 'left' or None.
python src/main.py --src ./data/trim10s.mp4 \
                   --model ./models/yolov5s-fp16.tflite \
                   --directions="{'total': None, 'inside': 'bottom', 'outside': 'top'}"
```

### Docker

```bash
./build_image.sh
./run.sh ./data/<YOUR_VIDEO_FILE>.mp4 ./models/<YOLOV5_MODEL>.tflite
```

Then you can see the results in `outputs` folder.


### Dataset
The example video on top of here is [TownCentreXVID](https://www.kaggle.com/ashayajbani/oxford-town-centre/version/4?select=TownCentreXVID.mp4).
You can download it from the link (kaggle).

I recoomend to trim it for about 10s because it's too big for testing.


## <div align="center">Citations</div>

### SORT

```
@inproceedings{Bewley2016_sort,
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  title={Simple online and realtime tracking},
  year={2016},
  pages={3464-3468},
  keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
  doi={10.1109/ICIP.2016.7533003}
}
```

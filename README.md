# Fall Detection with YOLOv8

Fall Detection with YOLOv8 is a computer vision project that aims to detect falls using the YOLOv8 object detection model. This project provides a real-time fall detection solution by analyzing video streams and triggering an alert when a fall is detected.

## Features

- Utilizes the YOLOv8 object detection model for accurate fall detection
- Real-time detection and immediate alert using visual cues
- Suitable for various scenarios including construction sites, low foot traffic areas, and monitoring the safety of the elderly at home
- Built with efficiency and ease-of-use in mind

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLOv8

## Getting Started

1. Clone the repository:

```
https://github.com/alijawad07/fall_detection_yolov8
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Update the configuration file with the appropriate paths and parameters.

4. Run the fall detection script:
```
python3 fall_detection.py --data --source --output --weights
```
- --data => .yaml file with dataset and class details

- --source => Path to directory containing video

- --output => Path to save the detection results

- --weights => Path to yolov8 weights file


## Acknowledgments

- Thanks to Roboflow for providing the comprehensive fall detection dataset used in training the YOLOv8 model.
- Special appreciation to Ultralytics for developing the YOLOv8 model and its integration with the project.

## References

- [YOLOv8](https://github.com/ultralytics/yolov5)
- [Roboflow](https://roboflow.com/)

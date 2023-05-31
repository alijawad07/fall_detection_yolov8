import cv2
import time
from ultralytics import YOLO
import argparse
import yaml
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Run fall detection on a video')
    parser.add_argument('--data', required=True, help='Path to YAML config file')
    parser.add_argument('--source', required=True, help='Path to the input video')
    parser.add_argument('--output', required=True, help='Path to save the output video')
    parser.add_argument('--weights', required=True, help='Path to checkpoint file')
    return parser.parse_args()

args = parse_args()

# Load the model
model = YOLO(args.weights)

# Create output directory if it does not exist
os.makedirs(args.output, exist_ok=True)

# Load dataset parameters from YAML config file
with open(args.data, 'r') as f:
    config = yaml.safe_load(f)
classes = config['names']

# Initialize the fall detection parameters
fall_threshold = 0.5  # Confidence threshold for fall detection

# Start the video capture
cap = cv2.VideoCapture(args.source)

# Get the current time
start_time = time.time()
frame_count = 0

# Get the video frame rate and dimensions
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_video = cv2.VideoWriter(os.path.join(args.output, "output.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Initialize the red flashing effect parameters
red_flash_interval = 10  # Number of frames for each red flash
red_flash_duration = 5  # Number of frames for each red flash duration

# Initialize fall detection flag and alert status
fall_detected = False

while True:
    # Capture the frame
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    results = model(frame)[0]

    # Get the detections as a list of dictionaries
    detections = results.boxes.data.tolist()

    # Iterate over the detections
    for detection in detections:
        xmin, ymin, xmax, ymax, confidence, class_idx = detection

        if confidence > fall_threshold and classes[int(class_idx)] == 'Fall':
            fall_detected = True
            break

    # If fall is detected, add red flashing effect and show detection rectangle with class label
    if fall_detected:
        if frame_count % (red_flash_interval + red_flash_duration) < red_flash_duration:
            frame = cv2.addWeighted(frame, 1, np.zeros(frame.shape, dtype=np.uint8), 0.5, 0)
            frame[:, :, 2] = np.maximum(frame[:, :, 2], 255)

        # Display the detection rectangle with class label
        color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        cv2.putText(frame, classes[int(class_idx)], (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the FPS
    end_time = time.time()
    fps = int(frame_count / (end_time - start_time))
    cv2.putText(frame, "FPS: {}".format(fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame to the output video
    out_video.write(frame)

    frame_count += 1
    fall_detected = False

# Release the video capture and output video objects
cap.release()
out_video.release()

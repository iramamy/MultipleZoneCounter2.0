# 🎯 Multiple Zone Counter 2.0 🎯

**Multiple Zone Counter 2.0** is an advanced version of the [MultipleZoneCounter](https://github.com/iramamy/MultipleZoneCounter). This tool leverages OpenCV's powerful capabilities for drawing regions of interest and integrates YOLOv8 for efficient tracking and counting within these regions.

## ✨ Features

- **🖌️ Region Drawing with OpenCV**: Utilize OpenCV to draw custom polygonal regions directly on video frames, providing flexibility and precision in defining areas of interest.
- **🔍 YOLOv8 Integration**: Employ YOLOv8 for robust object detection, tracking, and counting within the defined regions.
- **🔧 Dynamic Region Management**: Easily move and adjust regions on the fly, ensuring accurate tracking and counting even in dynamic environments.
- **⏱️ Real-time Processing**: Perform real-time object detection, tracking, and counting, making it suitable for applications requiring immediate feedback.

## 🚀 Getting Started

### 📥 Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/iramamy/MultipleZoneCounter2.0
    cd MultipleZoneCounter2.0
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### ▶️ Usage

1. Run the main script with the desired parameters:

    ```bash
    python yolo_counter.py
    ```

2. Use the mouse to draw polygonal regions on the first frame of the video. Press `p` to proceed to video playback.

3. During playback, the defined regions will dynamically track and count objects using YOLOv8.

### ⌨️ Keyboard Controls

- **🖱️ Left Mouse Button**: Draw points for the polygonal region.
- **🔲 Enter**: Finish drawing the current region.
- **🔄 R**: Remove the last drawn point.
- **💾 S**: Save all drawn regions and switch to playback mode.
- **⏯️ P**: Pause and resume playback.
- **🎯 T**: Start the model for tracking.
- **❌ Q**: Quit the application.

## Code
```python
import cv2
from shapely.geometry import Polygon

from collections import defaultdict
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

from Helper.utils import Color, draw_polygon


track_history = defaultdict(list)
current_region = None
zones = []
drawing_finished = False
current_zone = 0
drawing_mode = True

# Mouse callback function to draw polygon
def draw(event, x, y, flags, param):
    global zones, current_zone, drawing_finished
    if drawing_mode and not drawing_finished:
        if event == cv2.EVENT_LBUTTONDOWN:
            if current_zone >= len(zones):
                zones.append([])
            zones[current_zone].append((x, y))
            print(f"Point drawn at: ({x}, {y})")

# Mouse callback function to move polygon
def mouse_callback(event, x, y, flags, param):
    global current_region

    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False

# Create a window
cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Ultralytics YOLOv8 Region Counter Movable", draw)

def run(
    weights="yolov8n.pt",
    source="./data.mp4",
    device="cpu",
    view_img=True,
    save_img=True,
    exist_ok=False,
    classes=None,
    line_thickness=2,
    track_thickness=2,
    region_thickness=2,
    start_tracking=False,
):
    global current_zone, drawing_mode, counting_regions
    vid_frame_count = 0
    counting_regions = []
    initial_color = Color()

    # Initialize color

    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    model = YOLO(f"{weights}")
    model.to("cuda") if device == "0" else model.to("cpu")

    names = model.model.names

    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.avi"), fourcc, fps, (frame_width, frame_height))

    # Read the first frame
    success, frame = videocapture.read()

    if success:
        # Display the first frame for drawing polygons
        print("Draw polygons on the first frame. Press 'p' when done.")
        
        while True:
            frame_copy = frame.copy()
            
            color_used = draw_polygon(zones, initial_color, frame_copy)

            # Display the frame with polygons
            cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame_copy)

            # Handle key presses
            k = cv2.waitKey(1) & 0xFF

            # Exit on 'q' key press
            if k == ord('q'):
                break

            # Proceed to playback on 'p' key press
            if k == ord('p'):
                break

            elif k == ord("r"):  # 'r' key to remove the last point
                if len(zones) > 0 and len(zones[current_zone]) > 0:
                    zones[current_zone].pop()
                    print("Last point removed!")
                
            elif k == 13:  # Enter key to finish drawing in the current zone
                print(f"Drawing finished for region {current_zone + 1}")

                # Start a new region
                if len(zones[current_zone]) > 0:
                    
                    region_state = {
                        "name": "YOLOv8 Polygon Region",
                        "polygon": Polygon(zones[current_zone]),
                        "counts": 0,
                        "dragging": False,
                        "region_color": color_used[current_zone],
                        "text_color": (255, 255, 255),
                    }

                    counting_regions.append(region_state)
                    print("ZONES", zones)
                    zones.append([])
                    current_zone += 1
                    drawing_finished = False

                    print(f"Drawing new region {current_zone}")

            # "S" for saving all
            elif k == ord('s'): 
                drawing_finished = True
                print("All Drawing finished")

                drawing_mode = False
                cv2.setMouseCallback("Ultralytics YOLOv8 Region Counter Movable", mouse_callback)


    # Start video playback after drawing is done
    print("Playing video. Press 'p' to pause and 'q' to quit.")

    while videocapture.isOpened():
        ret, frame = videocapture.read()

        # Check if the video has ended
        if not ret:
            break

        vid_frame_count += 1

        if start_tracking:
            results = model.track(frame, persist=True, classes=classes)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()

                annotator = Annotator(frame, line_width=line_thickness, example=str(names))

                for box, track_id, cls in zip(boxes, track_ids, clss):
                    annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                    bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

                    track = track_history[track_id]  # Tracking Lines plot
                    track.append((float(bbox_center[0]), float(bbox_center[1])))
                    if len(track) > 30:
                        track.pop(0)
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                    # Check if detection inside region
                    for region in counting_regions:
                        if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                            region["counts"] += 1

            # Draw regions (Polygons/Rectangles)
            for region in counting_regions:
                region_label = str(region["counts"])
                region_color = region["region_color"]
                region_text_color = region["text_color"]

                polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
                centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

                text_size, _ = cv2.getTextSize(
                    region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
                )
                text_x = centroid_x - text_size[0] // 2
                text_y = centroid_y + text_size[1] // 2
                cv2.rectangle(
                    frame,
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    region_color,
                    -1,
                )
                cv2.putText(
                    frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
                )
                cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)

            if save_img:
                video_writer.write(frame)

            for region in counting_regions:  # Reinitialize count for each region
                region["counts"] = 0

        if view_img:
            cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)


        # Handle key presses
        k = cv2.waitKey(1) & 0xFF

        # Exit on 'q' key press
        if k == ord('q'):
            break

        # Pause on 'p' key press
        if k == ord('p'):
            print("Paused. Press 'p' to resume.")
            while True:
                k = cv2.waitKey(1)
                if k == ord('p'):
                    break

        # "T" to start tracking
        elif k == ord("t"):
            print("Start tracking!")
            start_tracking = True

    del vid_frame_count
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()
        


if __name__ == "__main__":
    run()
```

## 🏷️ Tags:
- Ultralytics
- Computer vision
- Object tracking
- OpenCV
- Real-time Tracking
- Machine Learning
- Image Analysis

## Credit idea
The idea comes from [Ultralytics](https://docs.ultralytics.com/guides/region-counting/)

## Data for test
[Pexels](https://www.pexels.com/search/videos/traffic/)
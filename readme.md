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

## 🏷️ Tags:
- Ultralytics
- Computer vision
- Object tracking
- OpenCV
- Real-time Tracking
- Machine Learning
- Image Analysis

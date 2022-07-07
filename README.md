# YoungWoo Global Learning - Final Project

1. Flask -opencv webcam / streaming server
2. Flask -mediapipe hand-detection
3. Flask -mediapipe face-detection

## Requirements

In order to execute the script you need to install opencv3 -> `import cv2` and some python modules.

There is a version that I made this project

opencv-python==4.6.0
Flask==2.1.2
mediapipe==0.8.9.1

In terminal, "python server.py"

than, the projects will run.

The project is default configured to run at port 5000. To change the running port you must specify the argument `-p, --port [PORT]`

### Change running port

The project is default configured to run at port 5000. To change the running port you must specify the argument `-p, --port [PORT]`.

_Example:_

`python3 server.py -p 3000 ` Runs on port 3000

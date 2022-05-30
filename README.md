# Vision system for identifying screws in smartphones


### Author

Nathan Wooster


### Description

This code allows an overhead camera to detect screw positions in a smartphone and output the coordinates to a robot for screw extraction.


### Dependencies

These common python libraries need to be installed. They are imported at the beginning of the scripts:

- numpy

- pandas

- cv2

- math

- matplotlib


### Installing
**The main scripts to be downloaded are located in:**

Run_vision_system

**The main script to be run is:**

main_vision.py

**The modules that also must be downloaded that it calls are:**

take_picture.py
calibrate_camera.py
screw_location.py




### Executing Program
To execute the script assign a camera in 'main_vision.py' and execute the script.
Other modules are included in the folder for tuning Hough Circle parameters using an error function.


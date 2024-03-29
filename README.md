# Detection, classification and segmentation of cracks in asphalt roads using artificial vision and convolutional neural networks

## Project Idea
Cracks in asphalt roads are one of the most important type of deterioration a road can face. They allow water to filter down to the base of the road, causing the deformation of the road and the appearance of potholes. Eventually, all these problems originated by cracks make the road virtually unusable. Therefore, detecting and repairing cracks is crucial to extend pavement's life. Unfortunately, nowadays, crack inspection is done manually, requiring a lot of effort and time. This project has the objective to improve and automate the visual inspection of asphalt roads by using convolutional neural networks and a GPS to detect, classify and locate where the cracks are in a map and easily show the information gathered by the system.

## Project overview

### Data Gathering

The data consist on a MP4 video of the route and a text file with the recording of the position of the vehicle every one second.

### Detection and classification

The model detects each crack that appears on video using YOLO, assigns a unique number, saves an image of the detection and in a database stores the moment at which the crack is detected.
The crack is followed along the video using the _Norfair Algorithm_, therefore the programme is able to uniquely identify each crack with a number.

Once the video is completely analysed, a database with all cracks that appear in the video is merged with a database with the position of the vehicle, therefore by using the time at which the crack was detected and obtaining where the vehicle was at that moment the programme is capable of determining where the crack is, geographically speaking.


### Segmentation



### Visualization


## How to clone


## Results


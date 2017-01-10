# denLabeler
Tools for making image labels

# classifierLabeler.py
* Tool for creating labels for images from video. Takes in video file, saves out images. Edit the text at the start of the file to change where files are read from and saved to
* Will rescale and crop images. Edit the text at the start of the file to change the location and output size

### Controls
* q: quit
* numberKey: class to save image as
* ': skip the current frame
* f: set to fast mode, 1 frame every second (based on the fps reported from the video file)
* s: set to slow mode, show more frames, set but the variable numJumpSlow (default is 2 frames)
* -: rewind, will delete the image

### Outputs
* Images to the folder specified in pathToImages. Images are named based on: `<original video name>-<frame number>-class<class number><image format>`
* imagePathsAndClass.txt: file listing the path to the image (as listed in pathToImages with the image file name) and the class. Is in the format so the  caffe tool convert_imageset will work with it.


# semanticLabeler.py

Semantic labeling tool built in python using openCV

Is a simple tool to do semantic labeling of video datasets, currently only supports single images.

Once the basic workflow is complete, will be adding a tool to interpolate to the next frame from the current segmentation map using optical flow in openCV.

# Controls 
* q: quit
* n: save segmentations, go to next frame
* p: Mode to create new polygons around objects
* e: Mode to edit polygons

### When editing polygons:
* LBM inside the polygon to select it
* RMB to deselect the polygon
* LBM and hold while moving mouse to move the a point, release to stop moving point
* CTRL+LMB will add a point to the current polygon at the selected location
* SHIFT + LMB will delete the closest point on the selected polygon
* CTRL + SHIFT + LMB inside selected polygon to delete it

# Settings
There are a bunch of hardcoded settings currently. Eventually they will be moved to an external config file. Read the settings class definition to see what they are set to. 

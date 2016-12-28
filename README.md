# denLabeler.py
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

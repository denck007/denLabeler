'''
Read in image from video, resize image, classify the image, save to txt file

image name format:
  <originalVideoName>-<number>-class<numberOfTheClass>.jpeg
folder Structure:
  base directory
    video
      video files (*.avi)
    images
      image files (*.jpeg)
      
'''

import cv2
import glob
import os
import numpy as np
from skimage import exposure


###
# Start to configure here
###
basePath = "/home/neil/AnnotationPrograms/Dataset/"
pathToVideo =  basePath + "vidTest/"
pathToFinishedVideos = pathToVideo + "done/"
pathToImages = "" # this is what prepends the listing in the output text file
imageListingPath = pathToFinishedVideos
imageListingFileBaseName = "_imagePathAndClass.txt"
labelFileName = basePath + "sysnet_words.txt"
imageBaseCropFileName = "_baseCrop.txt"

videoExtension = '*.avi'
imageExtension = '.jpeg'

saveImages = False


bottomCorner = [480,255] #down, right (from top left corner)
imgSize = [150,400] #height, width
newSize = [150,400]

labels = {0:'a',
          1:'b',
          2:'c',
          3:'d',
          4:'e',
          5:'f',
          6:'g',
          7:'h',
          8:'i',
          9:'j'}

# The number of frames to skip when in slow mode
numJumpSlow = 2

###
# End of configuration
###

videoNames = glob.glob(pathToVideo+videoExtension)

def setRegion(cap,bottomCorner,imgSize):
  # allow each video to have a different crop
  # user moves the window location for each video at the start
  
  baseCrop = [bottomCorner[0]-imgSize[0],bottomCorner[0],bottomCorner[1],bottomCorner[1] + imgSize[1]]
  
  currentEditPoint = 0
  winName = 'Set Region'
  cv2.namedWindow(winName)
    
  ret, rawFrame = cap.read()
  maxWidth = cap.get(3)
  maxHeight = cap.get(4)
  
  if ret:
    print('\n\nUse arrow keys to change window location')
    print('Use tab key to move between points')
    print('Enter q when done\n\n')
    while True:
      #rawFrame[bottomCorner[0]-imgSize[0]:bottomCorner[0],bottomCorner[1]:bottomCorner[1]+imgSize[1],:]
      img = rawFrame[baseCrop[0]:baseCrop[1],baseCrop[2]:baseCrop[3]]
      cv2.imshow(winName, img)
      
      key = cv2.waitKey(0) & 0xFF
      
      if key == ord('q'):
        cv2.destroyWindow(winName)
        return ([baseCrop[1],baseCrop[2]], [baseCrop[1]-baseCrop[0], baseCrop[3]-baseCrop[2]])
      if key == 9: # tab, allow the user to jump between top left (0) and bottom right (1) points
        if currentEditPoint == 0:
          currentEditPoint = 1
        else:
          currentEditPoint = 0
      
      if key == 81: # left arrow
        baseCrop[currentEditPoint+2] = max(baseCrop[currentEditPoint+2] - 1, 0) # dont let it go zero
      if key == 83: # right arrow
        baseCrop[currentEditPoint+2] = min(baseCrop[currentEditPoint+2] + 1, maxWidth)
      if key == 82: # up arrow
        baseCrop[currentEditPoint] = max(baseCrop[currentEditPoint] - 1, 0)
      if key == 84: # down arrow
        baseCrop[currentEditPoint] = min(baseCrop[currentEditPoint] + 1, maxHeight) 
      
  
# save out a file that describes what the class labels mean
labelText = ""
for l in labels:
  labelText = labelText + str(l) + " " + labels[l] + "\n"

labelFile = open(labelFileName,"w+")
labelFile.write(labelText)
labelFile.close()



for vid in videoNames:
  printNameCounter = 0 #need this incase something happens, we still know what video we are on
  cap = cv2.VideoCapture(vid)
  numFrames = cap.get(7)
  numFramesLen = len(str(numFrames))
  fps = cap.get(5)
  videoLength = numFrames/fps
  print("Video: " + vid)
  print "Video is %0.1d seconds long, number of frames: %.0f, and runs at %0.1f fps" %(videoLength, numFrames, fps)
  
  # initialize the counters for the video
  currentFrame = 0
  numFramesToJump = int(fps)
  
  # these track the frames that were looked at for the video and what the name of each is
  # it is useful for when we have to rewind
  frameHistory = []
  frameHistoryName = []
  
  imageListingFile =  imageListingPath + vid[vid.rfind('/')+1:-len(videoExtension)+1] + imageListingFileBaseName
  
  # make sure the image listing file exists
  if os.path.isfile(imageListingFile):
    print('imageListingFile exists')
  else:
    f = open(imageListingFile,"w+")
    f.close()
    
  (bottomCorner, imgSize) = setRegion(cap, bottomCorner, imgSize)
  baseCropFile =  imageListingPath + vid[vid.rfind('/')+1:-len(videoExtension)+1] + imageBaseCropFileName
  #f = open(baseCropFile,"w+")
  #f.write('bottomLeft,{}\n'.format(bottomCorner))
  #f.write('imgSize,{}'.format(imgSize))
  #f.close()
  np.savetxt(baseCropFile,(bottomCorner,imgSize))
  

  # loop over every frame
  while (cap.isOpened() & (currentFrame < numFrames)): 
    cap.set(1,currentFrame)
    ret, rawFrame = cap.read()
    rawFrame = rawFrame.astype(float)/255.0
    
    
    # print the video name every 100 lines
    if printNameCounter<100:
      printNameCounter +=1
    else:
      print('Current Video: ' + vid)
      printNameCounter = 0
    
    if ret == True: # verify that something is returned
      croppedFrame = rawFrame[bottomCorner[0]-imgSize[0]:bottomCorner[0],bottomCorner[1]:bottomCorner[1]+imgSize[1],:]
      #cv2.imshow('Cropped Frame',croppedFrame)
      frame = cv2.resize(croppedFrame,(newSize[1],newSize[0]))
      cv2.imshow('Classify This Frame', frame)
      
      cv2.imshow('equalize_adapthist', exposure.equalize_adapthist(frame,kernel_size=None, clip_limit=0.01))
      #rescalePercentile = 3
      #p2, p98 = np.percentile(frame, (rescalePercentile,100-rescalePercentile))
      #cv2.imshow('rescale_intensity',exposure.rescale_intensity(frame, in_range=(p2, p98)))
      
      # wait for user to enter a key
      key = cv2.waitKey(0) & 0xFF
      keyChr = chr(key)
      
      # Verify that the key entered is a digit and that it is in the labels
      if keyChr.isdigit() and int(keyChr) in labels:
        currentLabel = keyChr
        print('{:.0f}% Complete'.format(currentFrame/numFrames*100) + '\t\tClassified as frame ' + str(currentFrame) + ' as: ' +labels[int(currentLabel)])
        
        # create the image file name
        frameNumberString = ('{0:0' + str(numFramesLen) + '}').format(currentFrame)
        classString = "-class" + currentLabel
        imgName = pathToImages + vid[vid.rfind('/')+1:vid.rfind('.')]+'-'+ frameNumberString + classString + imageExtension
        
        # save the image
        if saveImages:
          cv2.imwrite(imgName,frame)
          
        # save the path and class to a text file for caffe tool to convert to database
        classFile = open(imageListingFile,'a+')
        classFile.write(imgName + " " + currentLabel+'\n')
        classFile.close()
        
        # Save the history
        frameHistory.append(currentFrame)
        frameHistoryName.append(imgName)
        
        # increment the frame
        currentFrame += numFramesToJump
        
      elif key == ord('`'):
        print('Skipping frame: ' + str(currentFrame))
        currentFrame += numFramesToJump
      elif key == ord('f'):
        print('Setting to fast mode: Will operate at 1 frame per second')
        numFramesToJump = int(fps)
        
      elif key == ord('s'):
        print('Setting to slow mode: Will operate on every ' + str(numJumpSlow) + 'nd frame')
        numFramesToJump = numJumpSlow
      
      elif key == ord('b'): # the video is bad, move it to the bad folder
        badPath = pathToVideo +'bad/'
        if not os.path.isdir(badPath):
          os.mkdir(badPath)
        os.rename(vid, badPath + vid[vid.rfind('/')+1:])        
        print("Moved video to /video/bad/\n\n")
        break

      elif key == ord('-'):
        
        if frameHistory == []:
          print('Cannot rewind, already at start of video.')
          continue
        
        
        if (currentFrame > 0):
          # rewind the current Frame
          currentFrame = frameHistory.pop()
          nameToDelete = frameHistoryName.pop()
          
          print('Rewind to frame: ' + str(currentFrame))
          
          # Read in the text file, find the name, delete the name and everything after, save the text file
          classFile = open(imageListingFile,'r+')
          text = classFile.read()
          classFile.close()          
          text = text[:text.rfind(nameToDelete)]
          classFile = open(imageListingFile,'w+')
          classFile.write(text)
          classFile.close()
          
          # delete the image
          if os.path.isfile(nameToDelete) and saveImages:
            os.remove(nameToDelete)
          else:
            print("Issues trying to delete: " + nameToDelete)
          
        else: #so we cannot rewind to negative frames
          print('Already at frame 0')
                
      elif key == ord('q'):
        print('Exiting Video...')
        break
        #keepWorking = False

  #finished with the video
  cap.release() # release the video once done with it
  # if we are over 90% of the way through the video, move it
  if (currentFrame/numFrames > .98):
    os.rename(vid,pathToFinishedVideos + vid[vid.rfind('/')+1:])
  else:
    # need to delete the lines from the imageListingFile file and delete the images
    print('\n\n\nWarning: Did not finish and move video')
    print('Make sure to remove images and lines for video:')
    print vid
    
  if key == ord('q'):
    print('Exiting Program...')
    break

cv2.destroyAllWindows()



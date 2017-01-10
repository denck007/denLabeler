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


###
# Start to configure here
###
basePath = "/home/neil/AnnotationPrograms/Dataset/"
pathToVideo =  basePath + ""
pathToImages = basePath + "images_227x227/"
pathToFinishedVideos = pathToVideo + ""

videoExtension = '*.avi'
imageExtension = '.jpeg'


bottomCorner = [480,255] #down, left
imgSize = [200,400] #height, width
newSize = [227,227]

labels = {0:'class0',
          1:'class1',
          2:'class2',
          3:'class3'}

# The number of frames to skip when in slow mode
numJumpSlow = 2

###
# End of configuration
###


videoNames = glob.glob(pathToVideo+videoExtension)

for vid in videoNames:
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
  
  # loop over every frame
  while (cap.isOpened() & (currentFrame < numFrames)):
    cap.set(1,currentFrame)
    ret, rawFrame = cap.read()
    
    if ret == True: # verify that something is returned
      croppedFrame = rawFrame[bottomCorner[0]-imgSize[0]:bottomCorner[0],bottomCorner[1]:bottomCorner[1]+imgSize[1],:]
      cv2.imshow('Cropped Frame',croppedFrame)
      frame = cv2.resize(croppedFrame,(newSize[0],newSize[1]))
      cv2.imshow('Classify This Frame', frame)
      
      # wait for user to enter a key
      key = cv2.waitKey(0) & 0xFF
      keyChr = chr(key)

      # Verify that the key entered is a digit and that it is in the labels
      if keyChr.isdigit() and int(keyChr) in labels:
        currentLabel = keyChr
        print('{:.0f}% Complete'.format(currentFrame/numFrames*100) + '\t\tClassified as frame ' + str(currentFrame) + ' as: ' +labels[0])
        
        # create the image file name
        frameNumberString = ('{0:0' + str(numFramesLen) + '}').format(currentFrame)
        classString = "-class" + currentLabel
        imgName = pathToImages + vid[vid.rfind('/')+1:vid.rfind('.')]+'-'+ frameNumberString + classString + imageExtension
        
        # save the image
        cv2.imwrite(imgName,frame)
        # save the path and class to a text file for caffe tool to convert to database
        classFile = open(pathToImages + 'imagePathsAndClass.txt','a+')
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

      elif key == ord('-'):      
        
        if (currentFrame > 0):
          # rewind the current Frame
          currentFrame = frameHistory.pop()
          nameToDelete = frameHistoryName.pop()
          
          print('Rewind to frame: ' + str(currentFrame))
          
          # Read in the text file, find the name, delete the name and everything after, save the text file
          classFile = open(pathToImages + 'imagePathsAndClass.txt','r+')
          text = classFile.read()
          classFile.close()          
          text = text[:text.rfind(nameToDelete)]
          classFile = open(pathToImages + 'imagePathsAndClass.txt','w+')
          classFile.write(text)
          classFile.close()
          
          # delete the image
          if os.path.isfile(nameToDelete):
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
    # need to delete the lines from the imagePathsAndClass.txt file and delete the images
    print('\n\n\nWarning: Did not finish and move video')
    print('Make sure to remove images and lines for video:')
    print vid
    
  if key == ord('q'):
    print('Exiting Program...')
    break

cv2.destroyAllWindows()



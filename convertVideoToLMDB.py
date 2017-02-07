# -*- coding: utf-8 -*-
"""
Image Multiplier- Take in video, make images, make variations on images, save to db

Method:
1) Read in txt file that list what the image name would be, similar to caffe converter format. Format must be:
  <sourceVideoName>-<frameNumber>-class<classNumber>.<imageExtension> <classNumber>
2) Based on the the dicts that define what the variations make the image, make variations
3) Save to database
4) Find a way to shuffle the database

"""
import sys
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import lmdb
import caffe
from time import time

class data:
  def __init__(self, labelTransform, transforms):
    
    # input settings
    self.pathToData = "/home/neil/AnnotationPrograms/Dataset/video/done/"
    self.videoFormat = ".avi"
    
    self.baseImageCropDefault = (480-150,480,255,255+400) # the base crop to be used if no other one is supplied
    self.newImageSize = (64,64) # dimensions to make the output image

    # database output settings    
    self.lmdb_path = '/home/neil/AnnotationPrograms/Dataset/train_lmdb_20170201/'    
    

    # find the data files we are going to look at
    self.dataFiles = glob.glob(self.pathToData + "*_imagePathAndClass.txt")
    self.baseCropFiles = glob.glob(self.pathToData + "*_baseCrop.txt")
    self.videoFiles = glob.glob(self.pathToData + "*.avi")
    self.numDataFiles = len(self.dataFiles)
    
    
    self.transforms = transforms
    self.labelTransform = labelTransform
    
    
    self.labels = {0:'a',
          1:'b',
          2:'c',
          3:'d',
          4:'e',
          5:'f',
          6:'g',
          7:'h',
          8:'i',
          9:'j'}
    
  def getDataStats(self):
    # this dict hold the names of the video files and a dict of the frame to class label       
    self.data = {}
    
    # go through the dataFiles and find out how much data we are making
    self.counts = {}
    self.counts['fileLines'] = np.zeros(self.numDataFiles) # number of lines in a file
    self.counts['fileImages'] = {} # number of images made from a file
    self.counts['totalImages'] =0 # total number of images made
    self.counts['outputClasses'] = np.zeros((self.numDataFiles,len(self.labels))) # number of images of each class out from each file
    self.baseImageCrop = {}
    
    

    dataFileCounter = -1
    for dataFile in self.dataFiles:
      dataFileCounter += 1
      print('\nGetting stats from: ' + dataFile)
      vidNameShort = dataFile[dataFile.rfind('/')+1:dataFile.rfind('_')]
      vidName = self.pathToData + vidNameShort + self.videoFormat
      
      #verify the video exists, and has a reasonable change of being named right:
      if os.path.isfile(vidName) and len(vidName) > len(self.pathToData):
        self.data[vidName] = {} # add the video to the dict
        self.counts['fileImages'][vidName] = 0
      else:
        print('Cannont find video: ' + vidName)
        return -1

      # get the baseCrop info
      # first need to find the exact path for the file
      for ii in xrange(len(self.baseCropFiles)):
        if vidNameShort in self.baseCropFiles[ii]:
          (bottomCorner, imgSize) = np.loadtxt(self.baseCropFiles[ii]).astype(np.int)
          self.baseImageCrop[vidName] = [bottomCorner[0]-imgSize[0],bottomCorner[0],bottomCorner[1],bottomCorner[1] + imgSize[1]]
        else: 
          print('No base crop file found for file {} using default values!'.format(vidName))
          self.baseImageCrop[vidName] = self.baseImageCropDefault
      
      #open the data file
      df = open(dataFile, "r+")
      for line in df:
        self.counts['fileLines'][dataFileCounter] += 1
        lineClass = int(line[line.rfind(' ')+1:-1]) # find the space, remove the new line
        lineFrame = int(line[line.find('-')+1:line.rfind('-')])
        
        # note that we are only counting here, we are not going to generate anything
        for transform in self.transforms:
          #print('vidName: ' + vidName + ' line ' + line + ' tType: ' + self.transType[tType][0])
          newClass = self.labelTransform[self.transforms[transform]['label']][lineClass]
          if newClass != -1:
            self.data[vidName][lineFrame] = lineClass
            self.counts['totalImages'] += 1
            self.counts['fileImages'][vidName] += 1
            self.counts['outputClasses'][dataFileCounter,newClass] += 1
      print('Making {} images from the original {} images in the video'.format(self.counts['fileImages'][vidName],self.counts['fileLines'][dataFileCounter]))
      df.close()

  def setMapsize(self):
    # find out how big to make the lmdb
    # make the db bigger by some multiple of the size needed for the entire db
    multiplier = 10
    numImagesInDB = self.counts['totalImages']
    self.map_size = int(np.prod(self.newImageSize)*3*numImagesInDB*multiplier) # size to set lmdb map_size to
    print('map_size: {}'.format(self.map_size))
    
  def transform(self, image, transformParams):
    # perform a transform on a given image
    # order is: find the crop, compute affine, crop the image, resize the image, mirror the image
  
    # get all the needed info out of the transform parameters
    if 'cropShiftRange' in transformParams:
      cropShift = np.random.random_integers(transformParams['cropShiftRange'][0] ,transformParams['cropShiftRange'][1])
      crop = [self.baseImageCrop[vid][0],self.baseImageCrop[vid][1], self.baseImageCrop[vid][2]+cropShift, self.baseImageCrop[vid][3]+cropShift]
    else:
      cropShift = 0
      crop = self.baseImageCrop[vid]
    # validate the crop
    maxWidth = image.shape[1]
    maxHeight = image.shape[0]
    assert crop[0] >= 0, 'cannot crop image, crop[0] not >= 0!'
    assert crop[1] <= maxHeight, 'cannot crop image, crop[1] not >= {}!'.format(maxHeight)
    assert crop[2] >= 0, 'cannot crop image, crop[2] not >= 0!'
    assert crop[3] <= maxWidth, 'cannot crop image, crop[3] not >= {}!'.format(maxWidth)
    
    if 'affineScaleRange' in transformParams:
      affineScale = np.random.uniform(transformParams['affineScaleRange'][0],transformParams['affineScaleRange'][1],1)
      doAffine = True
    else:
      affineScale = 0
      doAffine = False
      
    if 'mirror' in transformParams:
      mirror = transformParams['mirror']
    else:
      mirror = False
      
    # package the paramaters used up to send back
    transformParams = {'affineScale':affineScale,
                       'crop': crop,
                       'mirror':mirror}
    
    # start the transform        

    # affine is kind of expensive, so only do it if actually needed
    if doAffine:
      width = self.baseImageCrop[vid][3]-self.baseImageCrop[vid][2] #find the width 
      middle = width/2+self.baseImageCrop[vid][2] # find the middle of the frame
      
      # set the original points, and new points
      # the points are a triangle from the bottom left, to bottom right to top middle of the baseImageCrop
      # the skew is done by moving the bottom corners in and out the affineScale amount on each side
      pts1 = np.float32([[crop[2],crop[1]],[crop[2]+width,crop[1]],[middle,crop[0]]])
      pts2 = np.float32([[crop[2]+affineScale*width,crop[1]],[crop[2]+width-affineScale*width,crop[1]],[middle,crop[0]]])
      
      # get the transformation matrix and compute the new image
      M = cv2.getAffineTransform(pts1,pts2)
      image = cv2.warpAffine(image,M,(image.shape[1],image.shape[0])) # do it in place
    
    # crop the image and resize it to the new size
    image = image[crop[0]:crop[1], crop[2]:crop[3]]
    
    # resize
    image = cv2.resize(image,self.newImageSize)
    
    # mirror
    if mirror:
      image = cv2.flip(image,1)
                            
    return (image, transformParams)
    

    
    
# list out how the labels transform when crops and mirrors happen
labelTransform = {}
labelTransform['original'] = {0:0,
                              1:1,
                              2:2,
                              3:3,
                              4:6,
                              5:7,
                              6:4,
                              7:5,
                              8:8,
                              9:-1}
labelTransform['mirror'] =  {0:0,
                              1:1,
                              2:2,
                              3:3,
                              4:6, # right becomes left
                              5:7, # right becomes left
                              6:4, # left becomes right
                              7:5, # left becomes right
                              8:8,
                              9:-1}
labelTransform['mirror_noClass0'] =  {0:0,
                              1:1,
                              2:2,
                              3:3,
                              4:6, # right becomes left
                              5:7, # right becomes left
                              6:4, # left becomes right
                              7:5, # left becomes right
                              8:8,
                              9:-1}

labelTransform['noClass0'] =  {0:-1, # already have a ton of these
                              1:1,
                              2:2,
                              3:3,
                              4:4,
                              5:5,
                              6:6,
                              7:7,
                              8:8,
                              9:-1}


# list out how the transforms are to be done
# for affine and crops, list ranges of skewness and +/- pixle ranges (respectively) to shift by
# all transfroms must have a 'label' key that exists in the labelTransform dictionary
transforms = {}
#transforms[0] = {'label':'original'}
transforms[1] = {'affineScaleRange': (-.05,.15), 'cropShiftRange': (-75,75), 'label':'original'}
transforms[2] = {'affineScaleRange': (-.05,.15), 'cropShiftRange': (-75,75), 'label':'noClass0'}
transforms[3] = {'affineScaleRange': (-.05,.15), 'cropShiftRange': (-75,75), 'label':'mirror_noClass0'}




imgData = data(labelTransform,transforms)
sys.path.append(imgData.pathToData) # add the path to the videos to the system path
imgData.getDataStats()
imgData.setMapsize()


fig = plt.figure()
debug = False
keepWorking = True

# set up the database
lmdb_env = lmdb.open(imgData.lmdb_path, map_size = int(imgData.map_size))
lmdb_txn = lmdb_env.begin(write = True)

print('\n\nGoing to make {} images, estimated time: {:.2f} seconds'.format(imgData.counts['totalImages'], 0.008612749791328157*imgData.counts['totalImages']))

startTime = time()
# loop over all the videos
for vid in imgData.data:
  if not keepWorking:
    break

  cap = cv2.VideoCapture(vid)
  print("Converting Video: " + vid)
  print("There are " + str(imgData.counts['fileImages'][vid]) + ' images being made from this video')
  batchSizeCounter = 0
  batchCounter = 0
  vidName = vid[vid.rfind('/')+1:]
  
  randOrder = np.random.permutation(imgData.counts['fileImages'][vid])
  
  frameCount = 0
  # note this plays the images in alphabetical order
  # fn is frame number
  for fn in imgData.data[vid]:
    if not keepWorking:
      break
    
    # set the frame number to the current frame, read it, and get the original class
    cap.set(1,fn)
    ret, rawFrame = cap.read()
    imgClass = imgData.data[vid][fn]
    
    
    if debug:    
      pltNum = 230 # the subplot size
      
    # go over all the transformation types
    for transform in transforms:
      batchSizeCounter += 1
      imgClassNew = labelTransform[transforms[transform]['label']][imgClass] # look up the new label class
      #print('New Class: ' + str(imgClassNew))
      # only continue if it is not an ignore class
      if imgClassNew != -1:      
        frameCount += 1 # frame counter for printing
        # display info
        if frameCount %100 == 0:
          print('On frame {} of {} \t {:0.1f}% complete'.format(frameCount, imgData.counts['fileImages'][vid], frameCount*1.0/imgData.counts['fileImages'][vid]*100))
          
          
        # do the image transform, pass in the transform dict
        (img, transformParams) = imgData.transform(rawFrame, transforms[transform])
        
        # make a useful name for the database, must be useful enough to recreate the frame with a little work
        # slightly more complicated because we are using random crops on stuff
        # append on a random number so the frames are in a random order
        frameName = '{}_{}_fn:{}'.format(randOrder[0],vidName,fn)
        randOrder = np.delete(randOrder,0) # remove the first item        
        for key in transformParams:
          frameName = frameName + '_{}:{}'.format(key,transformParams[key])
        #for key in transforms[transform]:
        #  if key != 'label': # dont need the label info in the string
        #    frameName = frameName + '_{}:{}'.format(key,transforms[transform][key])
        
        if debug:
          pltNum += 1
          print('frame: {}\toriginalClass: {}\tnewClass: {}'.format(fn, imgClass, imgClassNew))          
          print(frameName)
          cv2.imshow('Image',rawFrame)
          plt.subplot(pltNum)
          plt.imshow(img)
          plt.show()
  
        img = img.transpose((2,0,1))
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 3
        datum.height = imgData.newImageSize[0]
        datum.width = imgData.newImageSize[1]
        datum = caffe.io.array_to_datum(img,imgClassNew)
        lmdb_txn.put(frameName,datum.SerializeToString())
        
        if (batchSizeCounter + 1) % 512 == 0:
          batchCounter += 1
          batchSizeCounter = 0
          lmdb_txn.commit()  # commit to the db
          lmdb_txn = lmdb_env.begin(write = True)
          print('Wrote batch {}'.format(batchCounter))
 
    if debug:
      key = cv2.waitKey(0) & 0xFF
      if key == ord('q'):
        keepWorking = False
        break    
      

  cap.release()
  # write the last batch to the db
  if (batchSizeCounter + 1) % 512 != 0:
    lmdb_txn.commit()  # commit to the db
    lmdb_txn = lmdb_env.begin(write = True)
    batchCounter = 0        
    print('Wrote last batch')

#lmdb_txn.commit()  # commit to the db
lmdb_env.close()   # close the database

plt.close()
cv2.destroyAllWindows()      

endTime = time()
totalTime = endTime-startTime
timePerFrame = totalTime/imgData.counts['totalImages']
print('Elapsed time to generate {} frames: {}'.format(imgData.counts['totalImages'], totalTime))
print('Averaged {} seconds/frame'.format(timePerFrame))




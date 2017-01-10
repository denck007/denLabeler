
import numpy as np
import cv2
import glob
import pickle

from time import time


def updateImage(s):
  
  
  if (time() - s.lastDraw) > 0.05:
    # this is to rate limit the drawing updates
    s.resetMasks()
    for poly in s.polygons:
      # Draw every polygon
      if poly.complete:
      #  print "Poly Complete"
        points = poly.pts.reshape((-1,1,2))
        cv2.fillPoly(s.polyMask,[points],poly.color)
      else: #not complete so draw the points
        drawPoints(s,poly)
        
      # draw the points if the polygon is complete AND the dots should be shown
      if poly.complete & s.dotShownAfterPolyDrawn:
        drawPoints(s,poly)
        
      
  
    #merge the dots and poly masks
    dotGray = cv2.cvtColor(s.dotMask,cv2.COLOR_BGR2GRAY)
    ret,mask = cv2.threshold(dotGray, 10, 255,cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    polyBG = cv2.bitwise_and(s.polyMask,s.polyMask,mask=mask_inv)
    dotFG = cv2.bitwise_and(s.dotMask,s.dotMask,mask = mask)  
    s.dst = cv2.add(polyBG,dotFG) #just put it on the polyMask as it is redrawn every time
  
    cv2.addWeighted(s.dst, s.alpha, s.img, 1-s.alpha, 0, s.dst)
    cv2.imshow(s.winName,s.dst)
    s.lastDraw = time()

def drawPoints(s,poly):
  # draw a point on the dotMask
  # Utility function so the same code is not repeated
  if poly.pts.shape[0] > 1: # test if there is a dot to be drawn
    points = poly.pts.reshape((-1,1,2)) # 
    for row in xrange(points.shape[0]): # iterate over every point
      cv2.circle(s.dotMask, (points[row,0,0],points[row,0,1]), s.dotSize , poly.color,-1)

    
  
  
class settings:
  'A spot to hold all the settings and references to stuff'  
  
  def __init__(self):
    #User settable stuff
    self.winName = "Labeler Window"
    self.alpha = 0.1    
    self.fileExtension = "*.jpeg"
    self.imgPath = "/home/neil/AnnotationPrograms/Dataset/"
    
    self.color = (255,255,0)
    self.dotSize = 6# how large the marker on a click is
    self.dotShownAfterPolyDrawn = True #is the dot on each vertex shown after the poly is drawn
    
    self.imgFolder = self.imgPath + "images/"
    self.groundTruthFolder = self.imgPath + "GT/"
    self.polygonListFolder = self.imgPath + "polygonLists/"
    self.imgNames = glob.glob(self.imgFolder+self.fileExtension)
    
    # Define stuff
    # create the window
    cv2.namedWindow(self.winName)
    cv2.setMouseCallback(self.winName,setPoint)
    
    self.selectedPolygon = -1
    self.selectedPoint = -1
    
    self.lastDraw = 0.0 # epoch of the last updateImage update
    
    self.labels = {'car':(0,255,255,0),'road':(1,255,0,0),'person':(2,0,255,255)}
    self.currentLabel = 'car'
    
  def loadImage(self,imgPath):
    'Load the image, initialize the masks, and create the list of polygons'
    self.img = cv2.imread(imgPath,1)
    self.resetMasks()
    
    # Set up the polygon lists for the image
    self.polygons = []
    self.polygons.append(polygon(self.currentLabel,self.labels[self.currentLabel][1:]))
    
  def resetMasks(self):
    'initialize and reset the masks used for drawing the polygons, dots, and the image to show'
    self.dotMask = np.zeros(self.img.shape,dtype=self.img.dtype)
    self.polyMask = np.zeros(self.img.shape,dtype=self.img.dtype)
    self.dst = np.zeros(self.img.shape,dtype=self.img.dtype)
    
  def incrementCurrentPoly(self):
    'increment the current polygon'
    print("Creating new polygon with label: " + self.currentLabel)
    self.polygons.append(polygon(self.currentLabel,self.labels[self.currentLabel][1:]))
    
def setPoint(event,x,y,flags,params):
  # stupid function so that cv2.setMouseCallback can call different items in an array
  # Tried to use cv2.setMouseCallback(self.winName,self.polygon[-1].setPoint) in settings.__init__() but, it never indexed through the polygons
  s.polygons[-1].setPoint(event,x,y,flags,params)

def savePolygonMask(imgName):
  outName = s.groundTruthFolder + imgName[len(s.imgFolder):-(len(s.fileExtension)-1)] + "_mask" + s.fileExtension[1:]
  cv2.imwrite(outName,s.polyMask)
  
def savePolygonList(imgName):
  outName = s.polygonListFolder + imgName[len(s.imgFolder):-(len(s.fileExtension)-1)] + "_List.pkl"
  output = open(outName,'wb',-1)
  pickle.dump(s.polygons,output)
  output.close()
  
class polygon:
  'This class defines a polygon'
  def __init__(self,label,color):    
    #print "initializing polygon"
    # initialize the pts array to impossible values
    self.pts = np.array([[-1,-1]],dtype=np.int32) 
    self.color = color
    self.label = label
    # flag for declaring if the polygon is done being defined
    self.complete = False
    

  def setPoint(self, event,x,y,flags,params):
  # This function draws a polygon based on where the user
  # Click the left button to set a pointe
  # Click the right button to end the poly, doesn't matter where you click
    if event == cv2.EVENT_RBUTTONUP:
      if self.pts.shape[0] >= 3: # is a complete polygon
        self.complete = True
        print("Finished creating polygon for label: " + self.label + " With color" + str(self.color))
        s.incrementCurrentPoly()
        
      updateImage(s)
    elif event == cv2.EVENT_LBUTTONUP:
      print (x,y)
      if self.pts[0,0] == -1:
          self.pts = np.array(np.array([x,y]).reshape(1,2))
      else:
          self.pts = np.append(self.pts,np.array([x,y]).reshape(1,2),axis=0)
      updateImage(s)

def editPolygonPoints(event,x,y,flags,params):
  'Callback for when in edit polygon mode'  
  
  # process for moving point:
  # 1. Click in polygon to select it
  # 2. click close to a point to select the point (hold mouse down)
  # 3. move the mouse to move the point
  # 4. release mouse to set the point
  # 5. edit other points in polygon
  # 6. RMB to release the polygon 
  
  # issue in python 2 where longs are not the same as ints
  flags = int(flags)
  
  # events for selecting/unselecting a polygon and point
  if event == cv2.EVENT_LBUTTONUP and s.selectedPolygon == -1:
    # user clicked to select a poly, no poly is currently selected
    s.selectedPolygon = selectPolygon(s.polygons,x,y)    
  elif event == cv2.EVENT_LBUTTONDOWN and s.selectedPolygon >= 0 and flags == (32):
    # user clicked and there is a polygon selected
    s.selectedPoint = findClosestPoint(s.polygons[s.selectedPolygon].pts,x,y)
  elif event == cv2.EVENT_LBUTTONUP and s.selectedPoint >= 0 and flags == cv2.EVENT_FLAG_LBUTTON + 32:
    # user released the button on the selected point
    # clear the point selection
    s.selectedPoint = -1
    updateImage(s)
  elif event == cv2.EVENT_RBUTTONUP:
    # user clicked the right button, they are done with this polygon
    s.selectedPolygon = -1
    
  # event for moving polygon
  elif event == cv2.EVENT_MOUSEMOVE and s.selectedPoint != -1:
    # a point has been selected and the mouse moved, update the point
    s.polygons[s.selectedPolygon].pts[s.selectedPoint,:] = [x,y]
    updateImage(s)
  
  # event for adding point
  elif event == cv2.EVENT_LBUTTONUP and s.selectedPolygon != -1 and flags == (cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_LBUTTON + 32):
    s.polygons[s.selectedPolygon].pts = insertPoint(s.polygons[s.selectedPolygon].pts,x,y)
    updateImage(s)
  
  #delete point  
  elif event == cv2.EVENT_LBUTTONUP and s.selectedPolygon != -1 and flags == (cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_LBUTTON + 32):
    print s.polygons[s.selectedPolygon].pts.shape[0]
    if s.polygons[s.selectedPolygon].pts.shape[0] >= 4: # make sure it can still be a polygon
      toDelete = findClosestPoint(s.polygons[s.selectedPolygon].pts,x,y)
      s.polygons[s.selectedPolygon].pts = np.delete(s.polygons[s.selectedPolygon].pts,toDelete,axis=0)
      updateImage(s)
  
  #delete polygon  
  elif event == cv2.EVENT_LBUTTONUP and s.selectedPolygon != -1 and flags == (cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_LBUTTON + 32):
    if cv2.pointPolygonTest(s.polygons[s.selectedPolygon].pts,(x,y),False) >= 0:
      s.polygons.pop(s.selectedPolygon)
      s.selectedPolygon = -1
      updateImage(s)
    
def insertPoint(pts,x,y):
  'Funtion inserts a point into the numpy array that defines the polygon'
  # find the closest point
  closestPoint = findClosestPoint(pts,x,y)
  
  numPoints = pts.shape[0]
  # find idx of neighboring points
  if closestPoint == numPoints:
    neighborPoints = [numPoints,1]
  elif closestPoint == numPoints:
    neighborPoints = [numPoints-1,0]
  else:
    neighborPoints = [closestPoint-1,closestPoint+1]
  
  
  insertAt = findClosestPoint(np.array([pts[neighborPoints[0],:],pts[neighborPoints[1],:]]),x,y) + closestPoint
  print("insertAt: " + str(insertAt) + " numPoints: " + str(numPoints))
  
  if insertAt < 0:
    return np.append(pts,insertAt,np.array([x,y]),axis=0)
  else:
    return np.insert(pts,insertAt,np.array([x,y]),axis=0)
  

  
  
  
def selectPolygon(polys,x,y):
  'determine if a point is inside a polygon, return the index of the polygon'
  
  for ii in xrange(len(polys)):
    if cv2.pointPolygonTest(polys[ii].pts,(x,y),False) >= 0:
      # is inside or on the edge of the polygon
      return ii
  return -1
  
def findClosestPoint(pts,x,y):
  'Find the point in the contour that is closest to the selected point'
  dist = np.square(pts[:,0] - x) + np.square(pts[:,1] - y)
  return np.argmin(dist)
    
s = settings()

quit = False
# loop over all the images in the folder
for imgName in s.imgNames:
  s.loadImage(imgName)
  # loop so that the window can be updated
  while True:
    updateImage(s)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("n"):
      # goto the next images
      # this will need to save stuff eventually
      savePolygonMask(imgName)
      savePolygonList(imgName)
      break
    if key == ord("e"): #edit
      # when changing modes, reset the selected polygon to none
      s.selectedPolygon = -1
      cv2.setMouseCallback(s.winName,editPolygonPoints)
    if key == ord("p"): #place points
      cv2.setMouseCallback(s.winName,setPoint)
    if key == ord("l"): #Change label
      userInput = raw_input("Enter class label: ")
      if userInput in s.labels:
        s.currentLabel = userInput
        s.color = s.labels[userInput][1:]
        
        # if we are in edit mode, change the color of the polygon
        if s.selectedPolygon != -1:
          s.polygons[s.selectedPolygon].color = s.color
        # the current polygon is not completed, but we are changing the color
        elif not s.polygons[-1].complete:
          s.polygons[s.selectedPolygon].color = s.color
          s.polygons[s.selectedPolygon].label = s.currentLabel
      else:
        print("Label " + userInput + " Is not defined. Available labels are:" + str(s.labels.keys()))
    elif key == ord("q"):
      print "Pressed q: Quitting..."
      quit = True
      break
      
  # test if the flag has been thrown to quit
  if quit:
    break

cv2.waitKey(100) & 0xFF
cv2.destroyWindow(s.winName)

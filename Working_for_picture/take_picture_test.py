import cv2 as cv
import time

#cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop)
cap = cv.VideoCapture(1) # video capture source camera (Here external webcam)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# ret,frame = cap.read() # return a single frame in variable `frame`

while (True):

    # Capture the video frame
    # by frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv.imshow('frame', frame)

    #

     #press 'q' button to exit video
    if cv.waitKey(1) & 0xFF == ord('q'):
       break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv.destroyAllWindows()

#while(True):
 #   cv2.imshow('img1',frame) #display the captured image
  #  if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
   #     cv2.imwrite('pics/c1.png',frame)
    #    cv2.destroyAllWindows()
     #   break

#cap.release()
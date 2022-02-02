import cv2 as cv
import time

# cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop)
cap = cv.VideoCapture(1)  # video capture source camera (Here external webcam)
print('camera connected')

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# ret,frame = cap.read() # return a single frame in variable `frame`

while (True):

    # Capture the video frame
    ret, frame = cap.read()
    # Display the resulting frame
    cv.imshow('frame', frame)
    pixel_size = frame.shape[:2]

    # press 'q' button to exit video
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

print(pixel_size)
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv.destroyAllWindows()

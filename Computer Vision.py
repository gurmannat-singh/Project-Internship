import cv2
cam = cv2.VideoCapture(0)                 #0 - INTERNAL , !-  EXTERANAL
rect,frame = cam.read()
cv2.waitKey(5)
cam.release()
cv2.destroyAllWindows()
print(rect)
print(frame)


cv2.namedWindow("Video Capturing",cv2.WINDOW_NORMAL)
cam = cv2.VideoCapture(0)
while True:
    rect,frame = cam.read()
    cv2.imshow("Video Capturing", frame)

    if cv2.waitKey(5)==ord("q"):
        break
cam.release()
cv2.destroyAllWindows()

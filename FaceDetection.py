import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smileCascade = cv2.CascadeClassifier("haarcascade_smile.xml")


# print(face_cascade)
def faceDetect(gFrame, cFrame):
    face = faceCascade.detectMultiScale(gFrame, 1.3, 5)
    for (x, y, w, h) in face:
        cv2.rectangle(cFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        newgFrame = gFrame[y:y + h, x:x + w]
        newcFrame = cFrame[y:y + h, x:x + w]
        eye = eyeCascade.detectMultiScale(newgFrame, 1.1, 3)
        # For smile detection
        # smile = smile_cascade.detectMultiScale(newgFrame, 1.7, 22)

        for (ex, ey, ew, eh) in eye:
            cv2.rectangle(newcFrame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # For smile detection
        # for (sx, sy, sw, sh) in smile:
        #     cv2.rectangle(newcFrame, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    return cFrame


videoCapture = cv2.VideoCapture(0)

while True:
    _, frame = videoCapture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = faceDetect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()

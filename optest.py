import cv2

cap = cv2.VideoCapture(0)

camera_parameter = ['CAP_PROP_FRAME_WIDTH',
 'CAP_PROP_FRAME_HEIGHT',
 'CAP_PROP_FOURCC',
 'CAP_PROP_BRIGHTNESS',
 'CAP_PROP_CONTRAST',
 'CAP_PROP_SATURATION',
 'CAP_PROP_HUE',
 'CAP_PROP_GAIN',
 'CAP_PROP_EXPOSURE',]

for x in range(9):
     print(camera_parameter[x], '=', cap.get(x))

while(True):
    ret, frame = cap.read()
    if not ret:
        print("No Frame")
    cv2.imshow("test frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow()
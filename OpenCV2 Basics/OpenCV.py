import cv2

# ----------------------------Image
# img = cv2.imread('Quotefancy.jpg')
# cv2.imshow('Image',img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
# -----------------------------Video
cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()

    if ret == False:
        continue
    cv2.imshow("Video Frame",frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


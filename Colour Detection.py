import numpy as np
import cv2

# Capturing video through webcam
webcam = cv2.VideoCapture(0)

# Define color ranges and kernel for dilation
color_ranges = {
    'Red': {'lower': np.array([136, 87, 111], np.uint8), 'upper': np.array([180, 255, 255], np.uint8), 'color': (0, 0, 255)},
    'Green': {'lower': np.array([25, 52, 72], np.uint8), 'upper': np.array([102, 255, 255], np.uint8), 'color': (0, 255, 0)},
    'Blue': {'lower': np.array([94, 80, 2], np.uint8), 'upper': np.array([120, 255, 255], np.uint8), 'color': (255, 0, 0)}
}

kernel = np.ones((5, 5), "uint8")

def detect_color(imageFrame, hsvFrame, color_name, color_info):
    mask = cv2.inRange(hsvFrame, color_info['lower'], color_info['upper'])
    mask = cv2.dilate(mask, kernel)
    res = cv2.bitwise_and(imageFrame, imageFrame, mask=mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), color_info['color'], 2)
            cv2.putText(imageFrame, f"{color_name} Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_info['color'])

while True:
    ret, imageFrame = webcam.read()
    if not ret:
        break

    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    for color_name, color_info in color_ranges.items():
        detect_color(imageFrame, hsvFrame, color_name, color_info)

    cv2.imshow("Multiple Color Detection in Real-Time", imageFrame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

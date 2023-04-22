import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # Open the camera

while True:
    ret, frame = cap.read()  # Capture a frame from the camera
    if not ret:
        break

    # Convert the frame to grayscale and apply a blur to reduce noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection to find the edges of the shapes
    edges = cv2.Canny(blur, 50, 150)

    # Find the contours of the shapes
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and draw a bounding box around each shape
    for cnt in contours:
        # Approximate the contour to reduce the number of points
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

        # Check the number of vertices to determine the shape
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
                shape = "square"
            else:
                shape = "rectangle"
        else:
            shape = "circle"

        # Draw the bounding box and label the shape
        cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 2)
        cv2.putText(frame, shape, (cnt[0][0][0], cnt[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show the resulting frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
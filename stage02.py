import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    # Convert the captured frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find barcodes in the grayscale frame
    barcodes = pyzbar.decode(gray)

    # Loop through each barcode in the frame
    for barcode in barcodes:
        # Extract the barcode's data and type
        data = barcode.data.decode("utf-8")
        barcode_type = barcode.type

        # Print the barcode's data and type on the terminal
        print("Type: " + barcode_type)
        print("Data: " + data)

    cv2.imshow("Barcode Scanner", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
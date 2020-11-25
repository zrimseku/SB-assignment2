import cv2
import os
import numpy as np


def camera_test():
    # Load the cascade
    ear_cascade = cv2.CascadeClassifier('cascade/cascade.xml')

    # To capture video from webcam.
    cap = cv2.VideoCapture(0)
    # To use a video file as input
    # cap = cv2.VideoCapture('filename.mp4')

    while True:
        # Read the frame
        _, img = cap.read()
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        ears = ear_cascade.detectMultiScale(img, 1.1, 100)
        for (x, y, w, h) in ears:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # Draw the rectangle around each ear
        # Display
        cv2.imshow('img', img)
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
    # Release the VideoCapture object
    cap.release()


def awe_test(cascade, scaleFactor, minNeighbours):
    results = {'IoU': 0, 'TP': 0, 'FP': 0}     # results will be averaged over all test images
    ear_cascade = cv2.CascadeClassifier(f'{cascade}/cascade.xml')
    for photo in os.listdir('AWEForSegmentation/test'):
        mask = cv2.imread(f'./AWEForSegmentation/testannot_rect/{photo}', 0) / 255
        img = cv2.imread(f'./AWEForSegmentation/test/{photo}', cv2.IMREAD_UNCHANGED)
        ears = ear_cascade.detectMultiScale(img, scaleFactor, minNeighbours)
        avg_iou = 0
        for (x, y, w, h) in ears:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw the rectangle around each ear
            intersection = mask[y:y+h, x:x+h]
            union = mask.copy()
            union[y:y+h, x:x+h] = 1
            iou = np.sum(intersection) / np.sum(union)
            avg_iou += iou
            if iou > 0.5:
                results['TP'] += 1
            else:
                results['FP'] += 1
        results['IoU'] += avg_iou / len(ears)
        if '3vf' in photo:  # False
        # if True:
            cv2.imshow('img', img)  # Display
            cv2.waitKey(0)
    return results


# camera_test()
print(awe_test('cascade', 1.1, 40))

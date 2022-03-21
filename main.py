import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('facevideos/pexels-ivan-samkov-6963299.mp4')

while True:
    success, img = cap.read()  # give us our frame
    cv2.imshow('image', img)
    cv2.waitKey(1)

    # u can see simple boilerplate of PROJECTS....

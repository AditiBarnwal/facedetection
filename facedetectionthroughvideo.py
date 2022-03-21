import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('facevideos/pexels-ivan-samkov-6963299.mp4')
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()  # give us our frame
    #  BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)  # process the rgb img and stored in results
    # now process this result ...we can print ths out we can write results
    # ths result can reduces the frame rate(fps) value

    # extract info
    if results.detections:
        for id, detection in enumerate(results.detections):
            # draw the points on face and rectangle also
            # mpDraw.draw_detection(img, detection)
            # now loop through each of results & can display them
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                        3, (255, 0, 0), 2)

    cv2.imshow('image', img)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    cv2.imshow('image', img)
    cv2.waitKey(1)  # if its fast then increase the value but will display actual value

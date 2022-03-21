import cv2
import mediapipe as mp
import time


class faceDetector():
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw= True):
        #  BGR to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)  # process the rgb img and stored in results
        # now process this result ...we can print ths out we can write results
        # ths result can reduces the frame rate(fps) value
        print(self.results)
        bboxs = []  # bounding boxes
        # extract info
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                self.mpDraw.draw_detection(img, detection)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bbox.append([id, bbox, detection.score])
                img = self.fancyDraw(img, bbox)
                cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                            3, (255, 0, 0), 2)

        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=10, rt=1 ):
        x, y, h, w = bbox
        x1, y1 = x+w, y+h
        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        cv2.line(img, (x, y), (x+l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        return img


def main():
    cap = cv2.VideoCapture('facevideos/pexels-ivan-samkov-6963299.mp4')
    pTime = 0
    detector = faceDetector()
    while True:
        success, img = cap.read()  # give us our frame
        # img, bboxs = detector.findFaces(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        cv2.imshow('image', img)
        cv2.waitKey(10)


if __name__ == '__main__':
    main()

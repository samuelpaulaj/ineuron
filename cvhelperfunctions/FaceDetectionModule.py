import cv2
import mediapipe as mp
import time # to display the framerate

class FaceDetector():
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxes = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # print(id, detection)
                # print(detection.score) # it will give the value of the score
                # print(detection.location_data.relative_bounding_box)
                # mpDraw.draw_detection(img, detection)
                bboxC = detection.location_data.relative_bounding_box #bboxC - bounding box class
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxes.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f"{int(detection.score[0]*100)}%", (bbox[0], bbox[1]-20),
                                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        return img, bboxes

    def fancyDraw(self, img, bbox, l = 30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y+h # defining bottom right corner point in the bounding box
        cv2.rectangle(img, bbox, (255, 0, 255), rt) # rt - rectangle thickness
        # top left x, y
        cv2.line(img, (x, y), (x+l, y), (255, 0, 255), t) # x and y is the starting point, l is the length we are defining
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # top right x1, y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255),t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # bottom left x, y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # bottom left x1, y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)


        return img




def main():
    cap = cv2.VideoCapture("Videos/2.mp4")
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxes= detector.findFaces(img)
        print(bboxes)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break

if __name__ == "__main__":
    main()
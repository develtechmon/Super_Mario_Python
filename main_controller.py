import cv2
import mediapipe as mp
import numpy as np
from time import sleep
import keyboard

class NoseHandTracking():
    def __init__(self, mode=False, upBody=False, smooth=True, detection=0.5, trackCon=0.5, maxHands=1):
        self.cap = cv2.VideoCapture(0)
        self.landmark = [0, 7, 8]
        self.img = 0
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detection
        self.trackCon = trackCon

        self.tipIds = [4, 8, 12, 16, 20]
        self.h = 0
        self.w = 0
        self.c = 0
        self.cx = 0
        self.cy = 0

        """Init Nose"""
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        """Init Hand"""
        self.maxHands = maxHands
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Initialize command
        self.command = None

    def findNoseHand(self, img, draw=False, NoseDetect=True, HandDetect=True): 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.resultsNose = self.pose.process(imgRGB)
        self.resultsHand = self.hands.process(imgRGB)

        if NoseDetect and draw and self.resultsNose.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.resultsNose.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        if HandDetect and self.resultsHand.multi_hand_landmarks:
            for handLms in self.resultsHand.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

    def findPosition(self, img, draw=True, NosePost=True, HandPost=True, handNo=0):
        lnList = []
        lhList = []

        if self.resultsNose.pose_landmarks:
            if NosePost:
                for id, lm in enumerate(self.resultsNose.pose_landmarks.landmark):
                    self.h, self.w, self.c = img.shape
                    self.cx, self.cy = int(lm.x * self.w), int(lm.y * self.h)
                    lnList.append([id, self.cx, self.cy, self.w, self.h])

                    if id in self.landmark:
                        cv2.circle(img, (self.cx, self.cy), 8, (0, 0, 255), cv2.FILLED)

            if HandPost and self.resultsHand.multi_hand_landmarks:
                myHand = self.resultsHand.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lhList.append([id, cx, cy])

                    if draw:
                        cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

        return [lnList, lhList]

    def fancyDraw(self, img, post):
        if len(post[0]) != 0 and post[0][0][0] == 0:
            cv2.circle(img, (post[0][0][1], post[0][0][2]), 6, (0, 255, 0), cv2.FILLED)

    def findPID(self, img, post):
        if len(post[0]) != 0 and post[0][0][0] == 0 and len(post[1]) != 0:
            fingers = []

            if post[1][self.tipIds[0]][1] < post[1][self.tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1, 5):
                if post[1][self.tipIds[id]][2] < post[1][self.tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            self.totalFingers = fingers.count(1)

            if fingers == [1, 1, 1, 1, 1]:
                self.command = "up"
                print("up")
            
            elif fingers == [1, 0, 0, 0, 0]:
                self.command = "right"
                print("right")

            elif fingers == [0, 0, 0, 0, 1]:
                self.command = "left"
                print("left")
                
            elif fingers == [1, 1, 0, 0, 0]:
                self.command = "jump right"
                print("jump right")

            elif fingers == [0, 0, 0, 1, 1]:
                self.command = "jump left"
                print("jump left")

            elif fingers == [0, 0, 0, 0, 0]:
                self.command = "down"
                
            else:
                self.command = "down"

        self.executeCommand()

    def executeCommand(self):
        # Release all arrow keys first
        keyboard.release("k")
        keyboard.release("l")
        keyboard.release("h")
        #keyboard.release("right")

        # Press the corresponding key
        if self.command == "up":
            keyboard.press("k")
                    
        elif self.command == "left":
            keyboard.press("h")
            
        elif self.command == "right":
            keyboard.press("l")
            
        elif self.command == "jump right":
            keyboard.press("k")
            keyboard.press("l")
            
        elif self.command == "jump left":
            keyboard.press("k")
            keyboard.press("h")
            
        elif self.command == "down":
            keyboard.release("k")
            keyboard.release("l")
            keyboard.release("h")   

if __name__ == "__main__":
    detect = NoseHandTracking()
    w, h = 640, 480

    while True:
        success, img = detect.cap.read()

        # Step 1 - Find Nose and Hands
        detect.findNoseHand(img)

        # Step 2 - Find Position
        post = detect.findPosition(img)
        
        # Step 3 - FancyDraw
        detect.fancyDraw(img, post)

        # Step 4 - Track
        detect.findPID(img, post)

        cv2.imshow("Streaming", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

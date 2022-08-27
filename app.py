import cv2
import mediapipe as mp
import time
from pynput import keyboard
from pynput.keyboard import Key, Controller
# import subprocess
import os

keyboard = Controller()

# automatically types the shortcuts on Toastify
# I will refer to these as hotkey functions


def play_pause():
    keyboard.press(Key.ctrl)
    keyboard.press(Key.alt)
    keyboard.press(Key.up)
    keyboard.release(Key.ctrl)
    keyboard.release(Key.alt)
    keyboard.release(Key.up)


def next_song():
    keyboard.press(Key.ctrl)
    keyboard.press(Key.alt)
    keyboard.press(Key.right)
    keyboard.release(Key.ctrl)
    keyboard.release(Key.alt)
    keyboard.release(Key.right)


def prev_song():
    keyboard.press(Key.ctrl)
    keyboard.press(Key.alt)
    keyboard.press(Key.left)
    keyboard.release(Key.ctrl)
    keyboard.release(Key.alt)
    keyboard.release(Key.left)
    time.sleep(.25)
    keyboard.press(Key.ctrl)
    keyboard.press(Key.alt)
    keyboard.press(Key.left)
    keyboard.release(Key.ctrl)
    keyboard.release(Key.alt)
    keyboard.release(Key.left)


def open_Spotify():
    keyboard.press(Key.ctrl)
    keyboard.press(Key.alt)
    keyboard.press('s')
    keyboard.release(Key.ctrl)
    keyboard.release(Key.alt)
    keyboard.release('s')

# This class determines hand type, puts a box around the hand, and writes whether L or R hand


class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                # lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                # bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                    bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                # draws everything (boxes around hands, writes which hand is showing)
                if draw:
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20,
                                   bbox[1] + bbox[3] + 20),
                                  (0, 225, 0), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 255, 0), 2)
                    if myHand["type"] == 'Left':
                        handType = 'left'
                        cv2.putText(img, 'L', (550, 100),
                                    cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 0), 8)
                    elif myHand["type"] == 'Right':
                        handType = 'right'
                        cv2.putText(img, 'R', (550, 100),
                                    cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 0), 8)
        # Here allows you to set draw to false (look at beginning of while loop to set it to false)
        if draw:
            return allHands, img
        else:
            return allHands


detector = HandDetector(detectionCon=0.8, maxHands=1)
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# using mediapipes coordinates
fingerCoordinates = [(8, 7), (12, 11), (16, 15), (20, 19)]
thumbCoordinate = (4, 3)
prevUpCount = 0  # previous number of fingers up
upCount = 0  # number of fingers up

# this determines the amount of time the same upCount needs to stay the same in order to be counted
timeForEachFinger = 0
fingerCount = 0  # upCount after timeForEachFinger determines it to be legit
handType = ''  # states the current hand L or R

# Opens Toastify which also opens Spotify
os.startfile('C://Program Files//Toastify//Toastify.exe')

# Starts loop with camera
while True:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    multiLandMarks = results.multi_hand_landmarks
    # whichhand = detector.findHands(img, draw=False) for without draw
    whichHand, img = detector.findHands(img)
    img = cv2.resize(img, (0, 0), fx=1, fy=1)

    # Start detection of fingers on hands
    if multiLandMarks:
        handPoints = []
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            for idx, lm in enumerate(handLms.landmark):
                # print(idx,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handPoints.append((cx, cy))

        # if a coordinates of fingers change then so does upCount
        for point in handPoints:
            cv2.circle(img, point, 5, (0, 0, 0), cv2.FILLED)
        for coordinate in fingerCoordinates:
            if handPoints[coordinate[0]][1] < handPoints[coordinate[1]][1]:
                upCount += 1
        if handPoints[thumbCoordinate[0]][0] > handPoints[thumbCoordinate[1]][0] and handType == 'right':
            upCount += 1
            # cv2.putText(img, '+THUMB', (100, 85),
            #             cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 8)
        elif handPoints[thumbCoordinate[0]][0] < handPoints[thumbCoordinate[1]][0] and handType == 'left':
            upCount += 1
            # cv2.putText(img, '+THUMB', (100, 85),
            #             cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 8)

        # if upCount stays the same after running through the loop 15 times then upCount turns into fingerCount
        if upCount == prevUpCount:
            if timeForEachFinger < 15:
                timeForEachFinger += 1
            elif timeForEachFinger == 15:
                fingerCount = upCount

                # certain amount of fingers run the functions that use the hotkey functions
                if fingerCount == 1:
                    play_pause()
                elif fingerCount == 2:
                    next_song()
                elif fingerCount == 3:
                    prev_song()
                elif fingerCount == 4:
                    open_Spotify()

                # prints fingerCount (mostly for testing)
                print(f"Finger Count: {fingerCount}")

                timeForEachFinger = 0  # resets timeForEachFinger

        # if upCount changes then time for each finger resets
        # Also prints on screen the last fingerCount and hotkey function used
        elif upCount != prevUpCount:
            timeForEachFinger = 0
            fingerCount = ""
        cv2.putText(img, str(fingerCount), (20, 100),
                    cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 0), 8)
        if fingerCount == 1:
            cv2.putText(img, "Play/Pause", (150, 75),
                        cv2.FONT_HERSHEY_PLAIN, 4, (0, 225, 0), 4)
        if fingerCount == 2:
            cv2.putText(img, "Next Song", (150, 75),
                        cv2.FONT_HERSHEY_PLAIN, 4, (0, 225, 0), 4)
        if fingerCount == 3:
            cv2.putText(img, "Last Song", (150, 75),
                        cv2.FONT_HERSHEY_PLAIN, 4, (0, 225, 0), 4)
        if fingerCount == 4:
            cv2.putText(img, "Open/Close", (125, 50),
                        cv2.FONT_HERSHEY_PLAIN, 4, (0, 225, 0), 4)
            cv2.putText(img, "Spotify", (200, 100),
                        cv2.FONT_HERSHEY_PLAIN, 4, (0, 225, 0), 4)

        # prints the three important variables (once again mostly for testing)
        print(f"{upCount} {fingerCount} {timeForEachFinger}")
        prevUpCount = upCount  # sets prevUpCount equal to the upCount that was just used
        upCount = 0  # resets upCount to 0

    cv2.imshow("Finger Counter", img)
    cv2.waitKey(1)
    # this allows you to press the 'X' on the top right of the webcam footage to close everything
    if cv2.getWindowProperty("Finger Counter", cv2.WND_PROP_VISIBLE) < 1:
        break
cv2.destroyAllWindows()

# closes Spotify which also closes Toastify
os.system('TASKKILL /F /IM Spotify.exe')

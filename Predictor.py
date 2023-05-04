import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from playsound import playsound
import threading
from google.protobuf.json_format import MessageToDict

class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.model = tf.keras.models.load_model('Classifier.h5')
        self.classes = ['Shambles', 'Room', 'Thousand years of death', 'Katon: Fireball', 'Rasengan', 'Mil flurur']
        self.message =""

    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        row = []
        if self.results.multi_hand_landmarks:
            
            if(len(self.results.multi_handedness) == 2):
                
                
                for handLms in self.results.multi_hand_landmarks:
                    for point in handLms.landmark:
                        row.append(point.x)
                        row.append(point.y)
                        row.append(point.z)

                    if draw:
                        self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
                #end for
               
            #end if
            else:
                for i in self.results.multi_handedness:
                    label = MessageToDict(i)['classification'][0]['label']
                    if(label == 'Left'):
                        for handLms in self.results.multi_hand_landmarks:
                            for point in handLms.landmark:
                                row.append(point.x)
                                row.append(point.y)
                                row.append(point.z)

                            if draw:
                                self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
                        #end for
                        for i in range(63):
                            row.append(0)
                    #end if
                    if(label=='Right'):
                        for i in range(63):
                            row.append(0)
                        for handLms in self.results.multi_hand_landmarks:
                            for point in handLms.landmark:
                                row.append(point.x)
                                row.append(point.y)
                                row.append(point.z)
                            
                            if draw:
                                self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
                        #end for
                    #end if
                #end for
            #end if
            if(len(row)>0):
                row_test = np.array(row).reshape(1,126)
                y_pred = self.model.predict(row_test)
                self.message = self.classes[np.argmax(y_pred)]
        #end if
        
        return image

    def positionFinder(self, image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
            # cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmlist

    def getMessage(self):
        return self.message

def play(file):
    playsound(file)
    

def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()
    counter = 0
    current, previous = "", ""
    while True:
        success, image = cap.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        current = tracker.getMessage()
        # Use putText() method for
        # inserting text on video
        cv2.putText(image,
                    tracker.getMessage(),
                    (50, 50),
                    font, 1,
                    (255, 24, 8),
                    2,
                    cv2.LINE_4)
        if(current == 'Room' and current != previous ):
            file = "Audio/trafalgar_law_room.mp3"
            t = threading.Thread(name='daemon', target=play, args=(file,))
            t.setDaemon(True)
            t.start()
        elif(current == 'Shambles' and current != previous):
            file = "Audio/trafalgar_law_shambles.mp3"
            t = threading.Thread(name='daemon', target=play, args=(file,))
            t.setDaemon(True)
            t.start()
        elif(current == 'Thousand years of death' and current != previous):
            file = "Audio/thousand.mp3"
            t = threading.Thread(name='daemon', target=play, args=(file,))
            t.setDaemon(True)
            t.start()
        elif(current == 'Katon: Fireball' and current != previous):
            file = "Audio/katon.mp3"
            t = threading.Thread(name='daemon', target=play, args=(file,))
            t.setDaemon(True)
            t.start()
        elif(current == 'Rasengan' and current != previous):
            file = "Audio/rasengan.mp3"
            t = threading.Thread(name='daemon', target=play, args=(file,))
            t.setDaemon(True)
            t.start()
        elif(current == 'Mil flurur' and current != previous):
            file = "Audio/robin.mp3"
            t = threading.Thread(name='daemon', target=play, args=(file,))
            t.setDaemon(True)
            t.start()
           
        previous = current
            
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)
        if len(lmList) != 0:
            
           
            print(lmList[4])
            print("_____________________________________")


        cv2.imshow("Gesture", image)
        #plt.show()

        if cv2.waitKey(20) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()

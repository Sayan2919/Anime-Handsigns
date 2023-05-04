import cv2
import mediapipe as mp
import csv
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
                row.append(5)
                with open('data/mir_flurur.csv','a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                    print(row)
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


def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()
    counter = 0
    while True:
        success, image = cap.read()
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)
        if len(lmList) != 0:
            counter += 1
            if(counter > 800):
                cv2.destroyAllWindows()
                break
            print(lmList[4])
            print("_____________________________________")




        cv2.imshow("Video", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()

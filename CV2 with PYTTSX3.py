import cv2
import pyttsx3

engine = pyttsx3.init()


def AI_speak(command):
    engine.say(command)
    engine.runAndWait()


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
classNames = []
classFile = 'coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

notificaiton_count = 0

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.7)

    try:
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                try:
                    label = classNames[classId - 1].upper()
                    confidence = round(confidence * 100, 2)
                    if label == 'PERSON' and confidence > 70:
                        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                        cv2.putText(img, label, (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(img, str(confidence), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                                    (0, 255, 0), 2)
                        print(label + " - " + str(confidence))
                        if notificaiton_count == 0:
                            AI_speak("Hello, how are you")
                            notificaiton_count += 1
                    elif notificaiton_count > 0:
                        AI_speak("Talk to you later")
                        notificaiton_count = 0

                        if confidence > 0.8:
                            print("hello")
                except IndexError:
                    pass
        else:
            pass
    except TypeError:
        pass

    cv2.imshow("Output", img)
    cv2.waitKey(1)

# importing the pyttsx library
import _thread
import threading

import pyttsx3
import cv2

engine = pyttsx3.init()


def onStart():
    print('starting')


def onWord(name, location, length):
    print('word', name, location, length)


def onEnd(name, completed):
    print('finishing', name, completed)


def speak(display):
    print("Starting speak")
    print(display)
    engine = pyttsx3.init()

    engine.connect('started-utterance', onStart)
    engine.connect('started-word', onWord)
    engine.connect('finished-utterance', onEnd)
    engine.say(display)
    engine.runAndWait()
    engine.stop()


def computerVision():
    thres = 0.7  # Threshold to check an object
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

    while True:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        print(classIds, bbox)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX,
                            1,
                            (0, 255, 0), 2)

                cv2.putText(img, str(confidence), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 0), 2)
                if confidence > 0.8:
                    try:
                        _thread.start_new_thread(speak, (str(classNames[classId - 1])))
                    except:
                        print("Error: unable to start thread")

        cv2.imshow("Output", img)

        cv2.waitKey(0)

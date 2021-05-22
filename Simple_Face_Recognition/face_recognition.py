# -*- coding: utf-8 -*-
"""
Created on Sat May 22 16:30:14 2021

@author: Dev
"""

# import os
import face_recognition
import cv2

# pip install cmake
# pip install dlib
# pip install face-recognition

def face_Rec(ipImg):
    try:
        image_to_be_matched = face_recognition.load_image_file(r"F:\myWork\pcControl\face_Rcognition\images\Dev.jpg")
        image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)[0]
        current_image = face_recognition.load_image_file(ipImg)
        current_image_encoded = face_recognition.face_encodings(current_image)[0]
        result = face_recognition.compare_faces(
            [image_to_be_matched_encoded], current_image_encoded)
        if result[0] == True:
            print ("Matched: " )
            return True
        else:
            print ("Not matched: ")
            return False
    except:
        print("except")
        return False
        



vid = cv2.VideoCapture(0)
currentframe = 0
while(True):
    ret,frame = vid.read()
    if ret:
        name = './logimg/cap.jpg'
        cv2.imwrite(name, frame)
        res=face_Rec(name)   
        if res == True:
            vid.release()
            break
    else:
        break
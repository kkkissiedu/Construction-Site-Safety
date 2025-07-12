from ultralytics import YOLO
import cv2
import cvzone
import math
import imageio

#_--------------For Webcam-----------------
#cap = cv2.VideoCapture(0)  #Object representing video capture device, 0 for internal webcam, 1 for external webcam
#Setting the resolution of the webcam
#cap.set(3, 1280)  #Set width
#cap.set(4, 720)  #Set height

#_--------------For Video File-------------
cap = cv2.VideoCapture('ppe.mp4')

if not cap.isOpened():
    print("Error: Could not open video file 'ppe.mp4'.")
    exit()

model = YOLO("PPE.pt")  #Loading trained YOLOv8 model for PPE detection


classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

frames = []
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Video FPS: {fps}")

myColour = (0, 0, 255)

while True:  #Infinite loop to continuously capture frames from the webcam
    success, img = cap.read()  #success will be 1 if frame was captured successfully, 0 otherwise, and the loop will end. Img will be a numpy array of the captured frame
    
    if not success:
        break

    results = model(img, stream=True)  #Perform inference on the captured frame, stream=True allows for real-time processing
    
    for r in results:
        boxes = r.boxes  #Return a list of detected bounding boxes

        for box in boxes:
            
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  #Convert coordinates from tensors to integers
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1
            bbox = x1, y1, w, h
            # cvzone.cornerRect(img, bbox)
            # cv2.rectangle(img, (x1, y1), (x2, y2), myColour, 2)

            #Confidence
            conf = math.ceil((box.conf[0]*100))/100
            
            #Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if conf > 0.5:
                if currentClass == 'Hardhat' or currentClass == 'Mask' or currentClass == 'Safety Vest':
                    myColour = (0, 255, 0)
                elif currentClass == 'NO-Hardhat' or currentClass == 'NO-Mask' or currentClass == 'NO-Safety Vest':
                    myColour = (0, 0, 255)
                else:
                    myColour = (255, 0, 0)
            
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', 
                               (max(0,x1), max(35, y1)), scale = 1, thickness = 1,
                               colorB = myColour,
                               colorT = (255, 255, 255),
                               colorR = myColour,
                               offset = 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), myColour, 2)

    height, width, _ = img.shape
    new_width = 640
    new_height = int((new_width / width) * height)
    img_resized = cv2.resize(img, (new_width, new_height))

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    frames.append(img_rgb)

imageio.mimsave('output_ppe.gif', frames, fps=15)
print("GIF saved successfully as 'output_ppe.gif'")

cap.release()
from cv2 import cv2
import numpy as np

#loading yolo

neuralNetwork = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')#you need to download those 

with open('coco.names', 'r') as names: #and this too 
    classNames = [x.strip() for x in names.readlines()]

layerNames = neuralNetwork.getLayerNames()
outLayers  = [layerNames[x[0]-1] for x in neuralNetwork.getUnconnectedOutLayers() ]
randColor  = np.random.uniform(0, 255, size=(len(classNames), 3))


#testing recognition video

camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()

    height, width, chanels = frame.shape
    # objects
    # geting features for diffrent color chanels
    features  = cv2.dnn.blobFromImage(frame, 0.00392, (320,320), (0,0,0),True, crop=False)

    neuralNetwork.setInput(features)
    objectInfo = neuralNetwork.forward(outLayers)

    #getting the confidence about object
    classIds, containers, assurances = [],[],[]
    for x in objectInfo:
        for y in x:
            results   = y[5:]
            classId   = np.argmax(results) 
            assurance = results[classId]
            if assurance > 0.5:
                centerX = int( y[0] * width  )
                centerY = int( y[1] * height )
                w       = int( y[2] * width  )
                h       = int( y[3] * height ) 
                x       = int(centerX - w / 2)
                y       = int(centerY - h / 2)
                

                containers.append([x, y, w, h])
                assurances.append(float(assurance))
                classIds.append(classId)
                
    indexes = cv2.dnn.NMSBoxes(containers, assurances, 0.5, 0,4)
    font    = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(containers)):
        if i in indexes:
            x, y, w, h = containers[i]
            name       = str(classNames[classIds[i]])
            color      = randColor[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y + 30), font,  1.3, color, 2)

    
    cv2.imshow("test video", frame)
    breakKey = cv2.waitKey(1)#for displaying every one milisecond
    if breakKey == 27:#ESC on keyboard
        break 

camera.release() 
cv2.destroyAllWindows()
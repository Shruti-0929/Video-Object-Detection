from itertools import count
import cv2
import numpy as np

cap = cv2.VideoCapture('Cut_Video.mp4')

min_height = 80
min_width = 80

count_line_position = 550

algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x,y,w,h):
    a=int(w/2)
    b=int(h/2)
    cx=x+a
    cy=y+b
    return cx, cy

detect = []
dot=6
count=0

video_stream = cv2.VideoCapture('Cut_Video.mp4')

total_frames=video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
print("Total Frames in video : ",total_frames)
frameCnt=0
while (frameCnt < total_frames-1):
  ret,frame = cap.read()
  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(3,3),5)

  img_sub = algo.apply(blur)
  dilat = cv2.dilate(img_sub,np.ones((5,5)))
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
  dilatada = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
  dilatada = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
  counterShape,h = cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  cv2.line(frame,(25,count_line_position),(1200,count_line_position),(255,120,0),3)



  for (i,c) in enumerate(counterShape):
      (x,y,w,h) = cv2.boundingRect(c)
      validate_counter = (w >= min_width) and (h >= min_height)
      if not validate_counter:
          continue
      cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,127),2) 
      #cv2.putText(frame,"Car",(x,y-20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))

      center = center_handle(x,y,w,h)
      detect.append(center)

      for (x,y) in detect:
          if x<(count_line_position+dot) and y<(count_line_position+dot):
               count+=1
               number=count
               detect.remove((x,y))
               
      
  cv2.imshow('Original Video',frame)
  frameCnt+=1
  if cv2.waitKey(1) == 13:
    break


print("Total Detected Vehicles : "+str(count))
cv2.destroyAllWindows()
cap.release()

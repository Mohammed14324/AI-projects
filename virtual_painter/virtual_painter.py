# get the distance between thumb and middle finger short to enable drawing

import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.9, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

global drawen,color,color_index,colors,w,h
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
drawen=set()
color=(0,0,0)
colors=["Black","Blue","Green","Red"]
color_index=0
def draw(cur_frame):
    global color ,color_index,colors
    cv2.rectangle(cur_frame,(0,0),(200,200),(0,0,0),3)
    cv2.putText(cur_frame,"Clear",(40,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),4)
    cv2.rectangle(cur_frame,(200,0),(400,200),(0,0,255),3)
    cv2.putText(cur_frame,"Red",(260,100),cv2.FONT_ITALIC,1,(0,0,255),4)
    cv2.rectangle(cur_frame,(400,0),(600,200),(255,0,0),2)
    cv2.putText(cur_frame,"Blue",(460,100),cv2.FONT_ITALIC,1,(255,0,0),4)
    cv2.rectangle(cur_frame,(600,0),(800,200),(0,255,0),2)
    cv2.putText(cur_frame,"Green",(660,100),cv2.FONT_ITALIC,1,(0,255,0),4)
    cv2.rectangle(cur_frame,(800,0),(1000,200),(0,0,0),2)
    cv2.putText(cur_frame,"Black",(860,100),cv2.FONT_ITALIC,1,(0,0,0),4)
    cv2.putText(cur_frame,colors[color_index],(int(w-100),100),cv2.FONT_ITALIC,1,color,4)
def get_pressed_button(x,y):
    global color ,color_index
    if(x>w-200 and y<200):
        if len(drawen):
            print("Cleared")
        drawen.clear()
        return True
    elif x<w-200 and x>w-400 and y<200:
        if color_index !=3:
            color=(0,0,255)
            color_index=3
            print("Color changed to red")
        return True
    elif x<w-400 and x>w-600 and y<200:
        if color_index !=1:
            print("Color changed to blue")
            color_index=1
            color=(255,0,0)
        return True
    elif x<w-600 and x>w-800 and y<200:
        if color_index !=2:
            print("Color changed to green")
            color_index=2
            color=(0,255,0)
        return True
    elif x<w-800 and x>x-1000 and y<200:
        if color_index !=0:
            print("Color changed to black")
            color_index=0
            color=(0,0,0)
        return True
    return False
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middel_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h),int(thumb_tip.z * w))
            index_coords = (int(index_tip.x * w), int(index_tip.y * h),int(index_tip.z * w))
            middel_coords = (int(middel_tip.x * w), int(middel_tip.y * h),int(middel_tip.z * w))
            dist = distance(thumb_coords, middel_coords)
            x=int(index_coords[0])
            y=int(index_coords[1])
            if(dist<100):
                if not get_pressed_button(x,y):
                    drawen.add((x,y,color))
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    for(x,y,col) in drawen:
        cv2.circle(frame,(x,y),30,col,-1)
    frame= cv2.flip(frame,1) 
    draw(frame)
    cv2.imshow("Finger Distance", frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()

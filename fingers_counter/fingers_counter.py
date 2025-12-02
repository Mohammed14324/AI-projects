import mediapipe as mp
import cv2
import warnings
def distance(point1,point2):
    return ((point1.x-point2.x)**2+(point1.y-point2.y)**2+(point1.z-point2.z)**2)**0.5
hands=mp.solutions.hands.Hands(max_num_hands=2,min_detection_confidence=0.8,min_tracking_confidence=0.5)
cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        break
    fingers=0
    results=hands.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame,hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS)
            thumb_mcp=hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_CMC]
            thumb_pip=hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP]
            thumb_tip=hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            index_mcp=hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
            index_pip=hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
            index_tip=hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            middle_mcp=hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
            middle_pip=hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
            middle_tip=hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_mcp=hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_MCP]
            ring_pip=hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_PIP]
            ring_tip=hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
            pinky_mcp=hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP]
            pinky_pip=hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_PIP]
            pinky_tip=hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
            landmark=hand_landmarks.landmark[0]
            if(distance(thumb_tip,thumb_pip)+distance(thumb_pip,thumb_mcp)+distance(thumb_mcp,landmark)*0.8<distance(landmark,thumb_tip)):
                fingers+=1
            if(distance(index_tip,index_pip)+distance(index_pip,index_mcp)+distance(index_mcp,landmark)*0.7<distance(landmark,index_tip)):
                fingers+=1
            if(distance(middle_tip,middle_pip)+distance(middle_pip,middle_mcp)+distance(middle_mcp,landmark)*0.7<distance(landmark,middle_tip)):
                fingers+=1
            if(distance(ring_tip,ring_pip)+distance(ring_pip,ring_mcp)+distance(ring_mcp,landmark)*0.7<distance(landmark,ring_tip)):
                fingers+=1
            if(distance(pinky_tip,pinky_pip)+distance(pinky_pip,pinky_mcp)+distance(pinky_mcp,landmark)*0.7<distance(landmark,pinky_tip)):
                fingers+=1
    
    frame=cv2.flip(frame,1)
    cv2.putText(frame,"Number of fingers raised: "+str(fingers),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),5)
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
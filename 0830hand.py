import cv2
import mediapipe
import math
import time

cap = cv2.VideoCapture(0)
cap.set(3,640) #設定解析度
cap.set(4,480) #設定解析度
m_drawing = mediapipe.solutions.drawing_utils
m_drawing_styles = mediapipe.solutions.drawing_styles
mpHands = mediapipe.solutions.hands
hands = mpHands.Hands()
mpDraw = mediapipe.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=10)
m_hands = mediapipe.solutions.hands
pTime = 0
cTime = 0
def vector_2d_angle(v1, v2): #根據acos計算兩個向量夾角
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle = 180
    return angle

# 根據手指關節的21個節點座標，計算每個手指的彎曲夾角
def hand_angle(hand):
    angles = []
    angle = vector_2d_angle( #大拇指角度
        ((int(hand[0][0])- int(hand[2][0])),(int(hand[0][1])- int(hand[2][1]))),
        ((int(hand[3][0])- int(hand[4][0])),(int(hand[3][1])- int(hand[4][1])))
        )
    angles.append(angle)

    angle = vector_2d_angle( #食指角度
        ((int(hand[0][0])- int(hand[6][0])),(int(hand[0][1])- int(hand[6][1]))),
        ((int(hand[7][0])- int(hand[8][0])),(int(hand[7][1])- int(hand[8][1])))
        )
    angles.append(angle)

    angle = vector_2d_angle( #中指角度
        ((int(hand[0][0])- int(hand[10][0])),(int(hand[0][1])- int(hand[10][1]))),
        ((int(hand[11][0])- int(hand[12][0])),(int(hand[11][1])- int(hand[12][1])))
        )
    angles.append(angle)

    angle = vector_2d_angle( #無名指角度
        ((int(hand[0][0])- int(hand[14][0])),(int(hand[0][1])- int(hand[14][1]))),
        ((int(hand[15][0])- int(hand[16][0])),(int(hand[15][1])- int(hand[16][1])))
        )
    angles.append(angle)

    angle = vector_2d_angle( #小拇指角度
        ((int(hand[0][0])- int(hand[18][0])),(int(hand[0][1])- int(hand[18][1]))),
        ((int(hand[19][0])- int(hand[20][0])),(int(hand[19][1])- int(hand[20][1])))
        )
    angles.append(angle)
    return angles

def hand_pos(angles): # 根據每根手指角度，判斷手勢
    f1 = angles[0]   # 大拇指
    f2 = angles[1]   # 食指
    f3 = angles[2]   # 中指
    f4 = angles[3]   # 無名指
    f5 = angles[4]   # 小指

    # 小於50表示手指伸直，大於等於50表示手指彎曲
    if f1<50 and f2>=50 and f3>=50 and f4>=50 and f5>=50:
        return 'good'
    elif f1>=50 and f2>=50 and f3>=50 and f4>=50 and f5>=50:
        return '0'
    elif f1>=50 and f2<50 and f3>=50 and f4>=50 and f5>=50:
        return '1'
    elif f1>=50 and f2<50 and f3<50 and f4>=50 and f5>=50:
        return '2'
    elif f1>=50 and f2>=50 and f3<50 and f4<50 and f5<50:
        return 'ok'
    elif f1<50 and f2>=50 and f3<50 and f4<50 and f5<50:
        return 'ok'
    elif f1>=50 and f2<50 and f3<50 and f4<50 and f5>50:
        return '3'
    elif f1>=50 and f2<50 and f3<50 and f4<50 and f5<50:
        return '4'
    elif f1<50 and f2<50 and f3<50 and f4<50 and f5<50:
        return '5'
    elif f1<50 and f2>=50 and f3>=50 and f4>=50 and f5<50:
        return '6'
    elif f1<50 and f2<50 and f3>=50 and f4>=50 and f5>=50:
        return '7'
    elif f1<50 and f2<50 and f3<50 and f4>=50 and f5>=50:
        return '8'
    elif f1<50 and f2<50 and f3<50 and f4<50 and f5>=50:
        return '9'
    else:
        return ''

cap = cv2.VideoCapture(0)            # 開啟WebCam
fontFace = cv2.FONT_HERSHEY_SIMPLEX  # 文字的字型
lineType = cv2.LINE_AA               # 文字的邊框

with m_hands.Hands( #啟用偵測手掌
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    if not cap.isOpened():
        print("無法開啟WebCam")
        exit()
    w, h = 540, 310
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        if not ret:
            print("無法擷取影格")
            break
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換色彩
        results = hands.process(img2)                # 偵測手勢
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = []                   # 記錄手指節點座標的串列
                for handLms in results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                for i in hand_landmarks.landmark:  # 將21個節點換算成座標
                    x = i.x*w
                    y = i.y*h
                    points.append((x,y))
                    print(i, x, y)
                    if i == 4:
                        cv2.circle(img, (x, y), 10, (0, 0, 255), cv2.FILLED)
                    print(i, x, y)
                if points:
                    angles = hand_angle(points)   # 計算手指角度
                    text = hand_pos(angles)            # 輸入手指的角度取得手勢名稱
                    cv2.putText(img, text, (30,120), fontFace, 4, (255,0,0), 5, lineType) # 印出手勢所表示意義

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow('hand', img)
        if cv2.waitKey(5) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

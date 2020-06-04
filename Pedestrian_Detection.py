
import cv2
import numpy as np

#video=cv2.VideoCapture(0)#需要更改为摄像头的地址
#cap = cv2.VideoCapture(0)#摄像头
#cap = cv2.VideoCapture(r'C:\Users\HP\Documents\Tencent Files\1398929343\FileRecv\MobileFile\VID20200423104144.mp4')#视频地址

def inside(r, q):#大方框嵌套小方框时将小方框移除
    rx, ry, rw, rh = r#大方框矩形
    qx, qy, qw, qh = q#小方框矩形
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):#绘矩形
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        #原版HOG中绘出的矩形和实际比例不符，这里缩小一下
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
def detect(img):#ret和img是通过ret，img=cap.read得来的两个参数
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    found, w = hog.detectMultiScale(img, winStride=(16, 16), padding=(32, 32), scale=1.01)
    found_filtered = []
    for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
            else:
                found_filtered.append(r)
    draw_detections(img, found)
    draw_detections(img, found_filtered, 3)
    return img

    #可以使用到主函数中
    #cv2.destroyAllWindows()
    #cap.release()
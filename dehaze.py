#对单帧图像处理的时候直接 J = DeHaze(frame) 即可，frame是输入的有雾图像矩阵，J是输出的无雾图像矩阵，DeHaze是去雾函数

import cv2
import math
import numpy as np

#用灰度直方图判断是否有雾以及雾的厚度,L=0表示没有雾，L=1表示轻雾，L=2表示浓雾
#无雾指能见度大于500m，轻雾指能见度在200m~500m之间，浓雾指能见度小于200m
def FogJudge(srcImg):
    a = [0, 0.0003, 0.0001, 0.0003, 0.002, 0.01, 0.01, 0.002, 0.00005, 0.0001, 0.005, 0.005]  #a为比例系数列表
    e = [0, 45, 5, 40, 0, 40, 0, 45, 5]                                                       #e为矫正范围列表
    T = [0, 90, 0.2, 30, 20, 20, 100, 0.18, 20,180]                                           #T为阈值列表
    L = 0                                                                                     #L为判断的结果，预设为0（无雾）

    gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
    [hight,width] = gray.shape

    #构建灰度直方图
    H = np.zeros(256,int)
    n = 0
    step = int(hight*width/10000)
    if step == 0:
        step = 1
    for j in range(0,hight,step):
        for i in range(0,width,step):
            H[gray[j][i]] = H[gray[j][i]] + 1
            n = n + 1

    #第一步，初步判断有没有雾
    h = np.zeros(256,int)
    for i in range(0,256):
        if H[i] <= a[1]*n:
            h[i] = 1
    hSum = np.sum(h)
    if hSum > T[1]:
        L = 1

    #如果初步判断没有雾，则进入第二步，对判断结果进行校准
    b1 = 0
    b2 = 0
    if L == 0:
        for i in range(0,256):
            if H[i] > n*a[2]:
                b1 = i
        N1 = 0
        for i in range(e[1],e[2]+1):
            N1 = N1 + H[b1-i]
        h2 = np.zeros(256, int)
        H2 = 0
        for i in range(0, 256):
            if H[i] > n * a[3]:
                b2 = i
            if H[i] < n*a[4]:
                h2[i] = 1
        for i in range(e[3],e[4]+1):
            H2 = H2 + h2[b2-i]
        if (float(N1/n) > T[2]) and (H2 > T[3]):
            L = 1

    #如果有雾，进入第三步，判断是轻雾还是重雾，如果是重雾，则令L=2
    if L == 1:
        b5 = 0
        for i in range(0, 256):
            if H[i] > n * a[8]:
                b5 = i
        N2 = 0
        for i in range(e[5], e[6] + 1):
            N2 = N2 + H[b5 - i]

        b6 = 0
        h6 = np.zeros(256, int)
        H6 = 0
        for i in range(0, 256):
            if H[i] > n * a[9]:
                b6 = i
            if H[i] < n * a[10]:
                h6[i] = 1
        for i in range(e[7], e[8] + 1):
            H6 = H6 + h6[b6 - i]

        b7 = 0
        for i in range(0, 256):
            if H[i] > n * a[11]:
                b7 = i
        b8 = 0
        for i in range(0,256):
            if H[i] > n * a[11]:
                b8 = i
                break
        H7 = b7 - b8
        if float(N2/n)>T[7] and H6 > T[8] and H7 < T[9]:
            L = 2

    return L

#基于移动模板的图像去雾法，用于除去轻雾
def applyMask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    #print('MASKED=', masked)
    return masked.filled()

def applyThreshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = applyMask(matrix, low_mask, low_value)
    #print('Low MASK->', low_mask, '\nMatrix->', matrix)

    high_mask = matrix > high_value
    matrix = applyMask(matrix, high_mask, high_value)

    return matrix


def simplestCb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0
    #print('HALF PERCENT->', half_percent)

    channels = cv2.split(img)
    #print('Channels->\n', channels)
    #print('Shape->', channels[0].shape)
    #print('Shape of channels->', len(channels[2]))

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)
        #print('vec=', vec_size, '\nFlat=', flat)
        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]

        #print("Lowval: ", low_val)
        #print("Highval: ", high_val)

        # 选出低于低门限和高于高门限的部分
        thresholded = applyThreshold(channel, low_val, high_val)
        # 标准化通道
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

#    以下是暗通道去雾，用于除去浓雾
#    去雾公式I(x) = J(x)*t(x) + A(1-t(x))

def DarkChannel(srcImage,sz):   #计算暗通道
    b,g,r = cv2.split(srcImage)
    dc = cv2.min(cv2.min(r,g),b)  #得到暗通道
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))  #定义一个矩形结构元素
    dark = cv2.erode(dc,kernel)   #对图像进行腐蚀，优化暗通道
    return dark


def EstimateAirlight(srcImage, minHight, minWidth):  #计算大气光A
    vAtom = [255, 255, 255]  # vAtom的三个元素表示大气光的rgb三个值，先预设为255
    srcImg = np.array(srcImage)
    hight = srcImg.shape[0]   #分别记录下原矩阵的高和宽，这里是三维矩阵
    width = srcImg.shape[1]
    while hight*width > minHight*minWidth:   #进行迭代，直到窗口大小小于设定的大小
        lrImg = np.array_split(srcImg, 2, axis=1)   # 将图像分成4份
        [Img11, Img12] = np.array_split(lrImg[0], 2, axis=0)
        [Img21, Img22] = np.array_split(lrImg[1], 2, axis=0)
        fourSection = [Img11, Img12, Img21, Img22]  # 将分割好的九个部分放入列表中，便于操作
        maxScore = 0
        score = 0
        for i in range(0,4):
            score = np.mean(fourSection[i])-np.std(fourSection[i])  #计算各部分的得分，得分score=平均值-标准差
            if score > maxScore:   # 用得分最高的那部分图像替代原图像
                maxScore = score
                srcImg = fourSection[i]
                [hight,width,_] = fourSection[i].shape
    nMinDistance = 0
    for nY in range(0,hight):    #取最亮的点作为大气光的值
        for nX in range(0,width):
            nDistance = math.sqrt(srcImg[nY][nX][0]*srcImg[nY][nX][0]+srcImg[nY][nX][1]*srcImg[nY][nX][1] + srcImg[nY][nX][2]*srcImg[nY][nX][2])
            if nMinDistance < nDistance:
                nMinDistance = nDistance
                vAtom[0] = srcImg[nY][nX][0]    #确定大气光
                vAtom[1] = srcImg[nY][nX][1]
                vAtom[2] = srcImg[nY][nX][2]
    #矫正大气光
    dif = [(int(vAtom[0])-int(vAtom[1]))*(int(vAtom[0])-int(vAtom[1])), (int(vAtom[0])-int(vAtom[2]))*(int(vAtom[0])-int(vAtom[2])),
           (int(vAtom[1])-int(vAtom[2]))*(int(vAtom[1])-int(vAtom[2]))]
    if dif[0]>dif[2] and dif[1]>dif[2]:
        vAtom[0] = (int(vAtom[1]) + int(vAtom[2]))/2
    elif dif[0]>dif[1] and dif[2]>dif[1]:
        vAtom[1] = (int(vAtom[0]) + int(vAtom[2]))/2
    elif dif[1]>dif[0] and dif[2]>dif[0]:
        vAtom[2] = (int(vAtom[0]) + int(vAtom[1]))/2
    vAtom = np.array([vAtom])
    return vAtom


def TransmissionEstimate(srcImage,A,sz):  #估算粗透射率
    omega = 0.95
    Ic = np.empty(srcImage.shape,srcImage.dtype)

    for ind in range(0,3):
        Ic[:,:,ind] = srcImage[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(Ic,sz)
    return transmission

def Guidedfilter(I,p,r,eps):             #导向滤波函数，I是引导图，p是需要滤波的图像，r为滤波半径，eps是设定好的常数
    #导向滤波公式为:q=a*p+b。p为输入,q为输出,a和b为根据I求出的常量；p,q,a,b都是矩阵
    #对原图像(p)、引导图(I)、原图像*引导图（I*p)分别做均值滤波
    mean_I = cv2.boxFilter(I,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(I*p,cv2.CV_64F,(r,r))

    #计算I*p的方差
    cov_Ip = mean_Ip - mean_I*mean_p

    #计算I的方差
    mean_II = cv2.boxFilter(I*I,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    #求导向滤波公式中的a和b
    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    #对a和b做均值滤波
    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    #导向滤波公式求输出
    q = mean_a*I + mean_b
    return q


def TransmissionRefine(srcImage,et,r,eps):           #用导向滤波器细化透射率  r为滤波半径
    gray = cv2.cvtColor(srcImage,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    t = Guidedfilter(gray,et,r,eps)    #将原图像的灰度图作为导向图，粗投射率图像作为待滤波图像，进行导向滤波
    return t


def Recover(srcImage,t,A,tx = 0.1):
    res = np.empty(srcImage.shape,srcImage.dtype)
    t = cv2.max(t,tx)

    for ind in range(0,3):
        res[:,:,ind] = (srcImage[:,:,ind]-A[0,ind])/t + A[0,ind]  #去雾公式
    return res


def DeHaze(frame,r=20,eps=0.85):            #总函数
    L = FogJudge(frame)
    if L ==0:
        J =frame
    elif L==1:
        J = simplestCb(frame, 1)
    elif L ==2:
        srcImage = frame.astype('float64') / 255
        dark = DarkChannel(srcImage, 15)   #计算暗通道
        A = EstimateAirlight(srcImage, 10, 10)  #计算大气光
        te = TransmissionEstimate(srcImage, A, 15)  #粗略估算透射率
        t = TransmissionRefine(frame, te,r,eps)  #细化透射率
        J = Recover(srcImage, t, A, 0.1)  #得到去雾图像
    return J


if __name__=='__main__':    #用于测试的主函数

    cap = cv2.VideoCapture('input2.mp4')
    while(True):
        ret, frame = cap.read()
        J = DeHaze(frame)
        cv2.imshow('Original Hazy',frame)
        cv2.imshow('Haze removed',J)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
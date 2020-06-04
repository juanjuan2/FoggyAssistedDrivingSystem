import tkinter as tk
from PIL import Image, ImageTk
import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
import dehaze
import Pedestrian_Detection


def main(video_path):
    return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    pb_file = "./yolov3_coco.pb"
    num_classes = 80  # 使用原有训练权重，有80个
    input_size = 416
    graph = tf.Graph()  # 计算图，表示实例化一个用于tensorflow计算和表示用的数据流图，不负责运行计算
    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)
    framenumber = 1

    with tf.Session(graph=graph) as sess:
        vid = cv2.VideoCapture(video_path)  # 获取视频
        while True:
            return_value, frame = vid.read()
            framenumber = framenumber + 1
            currentFrame = framenumber
            if currentFrame % 6 != 0: continue
            if return_value:
                J = dehaze.DeHaze(frame)
                # frame = cv2.cvtColor(J, cv2.COLOR_BGR2RGB)
                #image = Image.fromarray(J)
                frame = J
            else:
                raise ValueError("No image!")
            frame1 = Pedestrian_Detection.detect(frame)    #识别行人
            frame_size = frame.shape[:2]
            image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]
            prev_time = time.time()

            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[1], return_tensors[2], return_tensors[3]],
                feed_dict={return_tensors[0]: image_data})

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
            # np.concatenate  numpy.concatenate((a1,a2,...), axis=0)函数。能够一次完成多个数组的拼接。其中a1,a2,...是拼接数组的名字
            # np.reshape(a,newshape,order='C')  a:数组——需要处理的数据  newshape:新的格式——整数或整数数组，如(2,3)表示2行3列。新的形状应该与原来的形状兼容，即行数和列数相乘后等于a中元素的数量。
            bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size,
                                             0.3)  # pred_bbox 预测的框架  frame_size  框架尺寸
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            image = utils.draw_bbox(frame1, bboxes)  # 绘制框
            #result = np.asarray(image)
            curr_time = time.time()
            exec_time = curr_time - prev_time
            #info = "time: %.2f ms" % (1000 * exec_time)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
        cv2.destroyAllWindows()

def FunctionCamera():
    video_path = 0
    main(video_path)
    return

def FunctionVideo():
    video_path = "./docs/images/3.mp4"
    main(video_path)
    return
def hit_me():
    tk.messagebox.showinfo(title='启动中',message='请稍等……')

def selectPath(self):
    path_ = tkinter.filedialog.askopenfilename()
    path_ = path_.replace("/", "\\\\")
    path.set(path_)

if  __name__=='__main__':    #用于测试的主函数
    pygame.mixer.music.load()    #载入音乐
    window = tk.Tk()  # 实例化
    window.title('雾天辅助驾驶系统')  # 命名窗口类名
    window.geometry('700x500')  # 设定窗口宽高
    window.resizable(width=False, height=False)  # 锁定窗口
    sw = window.winfo_screenwidth()
    # 得到屏幕宽度
    sh = window.winfo_screenheight()
    # 得到屏幕高度
    ww = 700
    wh = 500
    x = (sw - ww) / 2
    y = (sh - wh) / 2
    window.geometry("%dx%d+%d+%d" % (ww, wh, x, y))
    bj = tk.Canvas(window,
                   width=700,  # 指定Canvas组件的宽度
                   height=500,  # 指定Canvas组件的高度
                   bg='white')  # 指定Canvas组件的背景色
    # 打开本地图片
    image = Image.open(r"背景.jpg")  # 这里替换成自己文件
    im = ImageTk.PhotoImage(image)

    bj.create_image(340, 250, image=im)
    bj.pack()  # 居中放置组件
    # 第一个按钮
    ConnectCamera = tk.Button(text='连接摄像头', bd='3', width='15', height='2', bg='Gray', font=('Arial', 20), fg='white',
                              activebackground='Khaki', command=FunctionCamera)
    # bg:背景颜色；bd:边框像素大小;font：字体，大小；fg：字体颜色；activebackground：按钮背景色
    ConnectCamera.place(x=240, y=240, anchor='w')  # 通过坐标放置
    # 第二个按钮
    CaptureVideo = tk.Button(text='输入视频', bd='3', width='15', height='2', bg='Gray', font=('Arial', 20), fg='white',
                             activebackground='Khaki', command=FunctionVideo)
    CaptureVideo.place(x=240, y=380, anchor='w')

    window.mainloop()  # 就相当于一个很大的while循环，每点击一次就会更新一次
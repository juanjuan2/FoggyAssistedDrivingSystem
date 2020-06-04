import tkinter as tk
import os
import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
import dehaze
import tkinter.filedialog
from PIL import Image, ImageTk
import threading
import winsound

def main(video_path):
    global end,init
    return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    pb_file = "./yolov3_coco.pb"
    num_classes = 80  # 使用原有训练权重，有80个
    input_size = 416
    graph = tf.Graph()  # 计算图，表示实例化一个用于tensorflow计算和表示用的数据流图，不负责运行计算
    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)
    framenumber = 1
    junzhi=10
    with tf.Session(graph=graph) as sess:
        vid = cv2.VideoCapture(video_path)  # 获取视频
        init = 0
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(os.path.join(path_,'result2.avi'), fourcc, 30.0, size)
        while (1):
            return_value, frame = vid.read()
            #frame = cv2.flip(frame, 1)   #水平颠倒
            framenumber = framenumber + 1
            currentFrame = framenumber
            if currentFrame % 6 != 0:
                out.write(frame)
                continue
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
            #prev_time = time.time()

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

            image,junzhi = utils.draw_bbox(frame1, bboxes, junzhi)  # 绘制框
            for i, bbox in enumerate(bboxes):
                coor = np.array(bbox[:4], dtype=np.int32)
                if coor[2]-coor[0] > junzhi*1.5:
                    winsound.Beep(600, 100)
            #pygame.mixer.music.play(1)
            out.write(image)
            #result = np.asarray(image)
            #cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            #result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            pilImage = Image.fromarray(image)
            pilImage = pilImage.resize((700, 380), Image.ANTIALIAS)
            tkImage = ImageTk.PhotoImage(image=pilImage)
            bj.create_image(0, 0, anchor='nw', image=tkImage)
            window.update_idletasks()
            window.update()
            str1.set('进行中')
            if end == 1:
                str1.set('欢迎使用')
                end = 0
                init = 1
                window.update()
                sess.close()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                str1.set('欢迎使用')
                window.update()
                init = 1

                break
    str1.set('欢迎使用')
    window.update()
    init = 1
    end = 0
    return

def FunctionCamera():
    global i
    str1.set('启动中，请稍等……')
    window.update()
    video_path = 0
    i = i+1
    main(video_path)
    return

def FunctionVideo():
    global i
    i = i+1
    str1.set('启动中，请稍等……')
    window.update()
    video_path = "./Rec.mp4"
    main(video_path)
    return

def thread_it(func, *args):
    # 创建线程
    t = threading.Thread(target=func, args=args)
    # 守护线程
    t.setDaemon(True)
    # 启动
    t.start()
def selectPath():
    global path_
    path_ = tk.filedialog.askdirectory()
    path.set(path_)
    return

def end():   #设置结束按钮
    global end
    end = 1
    return


if __name__=='__main__':    #用于测试的主函数
    global init,i
    i = 0
    init = 1
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
                   height=380,  # 指定Canvas组件的高度
                   bg='white')  # 指定Canvas组件的背景色
    bj.pack()  # 居中放置组件
    # 打开本地图片
    image = Image.open(r"背景.jpg")  # 这里替换成自己文件
    im = ImageTk.PhotoImage(image)
    if init == 1:
        bj.create_image(340, 190, image=im)

    # 第一个按钮
    ConnectCamera = tk.Button(text='连接摄像头', bd='3', width='10', height='2', bg='Gray', font=('Arial', 15), fg='white',
                              activebackground='Khaki', command=lambda : thread_it(FunctionCamera))
    # bg:背景颜色；bd:边框像素大小;font：字体，大小；fg：字体颜色；activebackground：按钮背景色
    ConnectCamera.place(x=10, y=450, anchor='w')  # 通过坐标放置
    # 第二个按钮
    CaptureVideo = tk.Button(text='输入视频', bd='3', width='10', height='2', bg='Gray', font=('Arial', 15), fg='white',
                             activebackground='Khaki', command=lambda :thread_it(FunctionVideo))
    CaptureVideo.place(x=160, y=450, anchor='w')
    path = tk.StringVar()
    label1 = tk.Label(window, text="目标路径:", width='10', height='1')
    label1.place(x=445, y=480, anchor='w')
    entry1 = tk.Entry(window, textvariable=path)
    entry1.place(x=510, y=480, anchor='w')
    button1 = tk.Button(window, text="路径", command=selectPath, width='4', height='1')
    button1.place(x=660, y=480, anchor='w')
    str1 = tk.StringVar()
    label = tk.Label(window, textvariable=str1, fg='grey', font=('Arial', 20))
    label.place(x=285, y=390, anchor='w')
    str1.set('欢迎使用')
    button2 = tk.Button(window,text='结束', bd='3', width='10', height='2', bg='Gray', font=('Arial', 15), fg='white',
                              activebackground='Khaki', command=end)
    button2.place(x=310,y=450,anchor='w')
    window.mainloop()  # 就相当于一个很大的while循环，每点击一次就会更新一次
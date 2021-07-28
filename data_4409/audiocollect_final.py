# coding:utf-8
# !/usr/bin/python

# Extract images from a bag file.

# PKG = 'beginner_tutorials'
import roslib  # roslib.load_manifest(PKG)
import rosbag
import rospy
import os
from bag_test import quaternion2euler,euler2quaternion,euler2rotation
# Reading bag filename from command line or roslaunch parameter.
# import os
# import sys

'''
class ImageCreator():
    def __init__(self):
        self.bridge = CvBridge()
        with rosbag.Bag('/home/kanghao/bagfiles/2019-02-19-15-23-09.bag', 'r') as bag:  # 要读取的bag文件；
            for topic, msg, t in bag.read_messages():
                if topic == "/usb_cam/image_raw":  # 图像的topic；
                    try:
                        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                    except CvBridgeError as e:
                        print(e)
                    timestr = "%.6f" % msg.header.stamp.to_sec()
                    # %.6f表示小数点后带有6位，可根据精确度需要修改；
                    image_name = timestr + ".png"  # 图像命名：时间戳.png
                    cv2.imwrite(rgb_path + image_name, cv_image)  # 保存；
                elif topic == "/scan":  # laser的topic；
                    timestr = "%.6f" % msg.header.stamp.to_sec()
                    laser_data_name = timestr + ".txt"
                    laser_data_path = os.path.join(laser_path, laser_data_name)
                    with open(laser_data_path, "w") as f:
                        f.write(str(msg))
'''
import wave
WAVE_OUTPUT_FILENAME = 'DATA/audio_X'
RATE = 16000
csv_filepath_train='DATA/audio_data_train1.csv'
csv_filepath_test='DATA/audio_data_test1.csv'
import numpy as np
import pandas as pd
import csv
import random

center_coord=np.array([2.4,3.2])
threshold =10
offset = 100
times=10
if __name__ == '__main__':
    writer_train = csv.writer(open(csv_filepath_train, 'w'))
    writer_train.writerow(['address','labels'])
    writer_test = csv.writer(open(csv_filepath_test, 'w'))
    writer_test.writerow(['address','labels'])
    with rosbag.Bag('kuailexiaoche8.bag', 'r') as bag:  # 要读取的bag文件；
        count=0
        #print(type(bag.read_messages()))
        framesbuf1 = []
        framesbuf2 = []
        framesbuf3 = []
        framesbuf0 = []
        framesbuf = []
        first_scip = 0
        for shift_flag in range(times):
            print(shift_flag)
            num_flag=0
            framesbuf1 = []
            framesbuf2 = []
            framesbuf3 = []
            framesbuf0 = []
            framesbuf = []
            first_scip = 0
            for topic, msg, t in bag.read_messages():
                first_scip+=1
                if first_scip < threshold:
                    continue
                #print(topic)
                #print('topic:', topic, '\n')
                if (topic =='/audio/channel1' or topic == '/audio/channel2' or
                                        topic == '/audio/channel3' or topic == '/audio/channel0'):

                    #print('topic:',topic,'\n')
                    #print('msg:',msg,'\n')
                    #print('t:',t,'\n')

                    framesbuf = np.fromstring(msg.data, dtype=np.short)
                    count += 1
                    # print(count)
                    if topic == '/audio/channel1':
                        framesbuf1.append(framesbuf)
                       # print(framesbuf)
                       # print(framesbuf1)
                        flag1 = 1
                        #print(framesbuf1)
                    if topic == '/audio/channel2':
                        framesbuf2.append(framesbuf)
                        flag2 = 1
                    if topic == '/audio/channel3':
                        framesbuf3.append(framesbuf)
                        flag3 = 1
                    if topic == '/audio/channel0':
                        framesbuf0.append(framesbuf)
                        flag0 = 1

                        #print(framesbuf2)
                    num_flag += 1
                    if num_flag < shift_flag*4+1:
                        framesbuf1 = []
                        framesbuf2 = []
                        framesbuf3 = []
                        framesbuf0 = []
                        framesbuf = []
                        count=0
                        continue
                    if count > (times*4-1):
                        if framesbuf1 == [] or framesbuf2 == [] or framesbuf3 == [] or framesbuf0 == [] or framesbuf == []:
                            print('error')
                            count = 0
                            framesbuf1 = []
                            framesbuf2 = []
                            framesbuf3 = []
                            framesbuf0 = []
                            framesbuf = []
                            frames = []
                            continue
                        #print(framesbuf1)
                        #print(count)
                        framesbuf0 = np.concatenate(framesbuf0)
                        framesbuf1 = np.concatenate(framesbuf1)
                        framesbuf2 = np.concatenate(framesbuf2)
                        framesbuf3 = np.concatenate(framesbuf3)
                        frames = np.array(list(zip(framesbuf0,framesbuf1, framesbuf2,framesbuf3))).flatten()
                        #print(frames)
                        if len(frames) < 4096*times:
                            print(len(frames))
                            count = 0
                            framesbuf1 = []
                            framesbuf2 = []
                            framesbuf3 = []
                            framesbuf0 = []
                            framesbuf = []
                            continue
                        # print(count)



                        for topic_t,msg_t,t_t in (bag.read_messages()):
                            #print(topic_t)
                            if t_t > t and (topic_t == '/nlink_linktrack_tagframe0'):

                                for topic_u,msg_u,t_u in (bag.read_messages()):

                                    if t_u > t_t and (topic_u == '/mobile_base/sensors/imu_data'):
                                    #print(msg.pos_3d)
                                        wf = wave.open(WAVE_OUTPUT_FILENAME + str(t), 'wb')
                                        wf.setnchannels(4)
                                        wf.setsampwidth(2)
                                        wf.setframerate(RATE)
                                        wf.writeframes(b''.join(frames))
                                        wf.close()
                                        count = 0
                                        framesbuf1 = []
                                        framesbuf2 = []
                                        framesbuf3 = []
                                        framesbuf0 = []
                                        framesbuf = []
                                        data = msg_u.orientation
                                        qua = np.array([data.x, data.y, data.z, data.w])
                                        euler = quaternion2euler(qua)[2]
                                        euler = euler - offset
                                        euler = euler*np.pi/180
                                        transform=np.array([[np.cos(euler), np.sin(euler)],
                                                            [-np.sin(euler),  np.cos(euler)]])
                                        #print(transform)
                                        #print(np.array(msg_t.pos_3d)[0:2])
                                        coord=np.matmul(transform,center_coord-np.array(msg_t.pos_3d)[0:2])/5
                                        csv_data = [(WAVE_OUTPUT_FILENAME + str(t)), coord]
                                        rnd=random.random()
                                        if rnd > 0.1:
                                            writer_train.writerow(csv_data)
                                        else:
                                            writer_test.writerow(csv_data)
                                        break
                                break

                    #dataframe = pd.DataFrame({'a_name': a, 'b_name': b})
        #dataframe.to_csv("audio_data.csv", index=False, sep=',')
                # FORMAT = Int16


        '''
        for topic, msg, t in bag.read_messages():
            count += 1
            print(topic,msg,t)
            if count > 200:
                break
        '''
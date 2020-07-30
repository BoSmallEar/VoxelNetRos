from cv_bridge import CvBridge
import numpy as np
import os
import cv2
import rospy
from sensor_msgs.msg import Image



if __name__ == '__main__': 
    rgb_image_root = '../data/rgb_image_0048' 
    file_list = []
    tmp_list = os.listdir(rgb_image_root)
    tmp_list.sort()
    for f in tmp_list:
        cur_file = os.path.join(rgb_image_root, f)
        file_list.append(cur_file)


    rospy.init_node('pub_rgb_image')
    pub_velo = rospy.Publisher("rgb_image", Image, queue_size=1)
    rate = rospy.Rate(0.2)
    bridge = CvBridge()

    pc_num_counter = 0
    while not rospy.is_shutdown():
        rospy.loginfo(file_list[pc_num_counter])

        # pc_data = np.fromfile(file_list[pc_num_counter], dtype=np.float32).reshape(-1, 4)
        cv_image = cv2.imread(file_list[pc_num_counter]) 
        image_message = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
        pub_velo.publish(image_message)
        # print(pc_data[0,0,:])
        #pc_data = pc_data[:, :, :4]
        #pc_data = np.fromfile(file_list[pc_num_counter], dtype=np.float32).reshape(-1, 4)


        pc_num_counter = pc_num_counter + 1
        if pc_num_counter >= len(file_list):
            pc_num_counter = 0
        rate.sleep()

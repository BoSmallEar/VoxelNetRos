import numpy as np
import rospy
import os

if __name__ == '__main__':
    test_root = '../data/npy_lidar_0048'
    # test_root = '/home/adam/data/voxel_net/KITTI/testing/velodyne'
    file_list = []
    tmp_list = os.listdir(test_root)
    tmp_list.sort()
    for f in tmp_list:
        cur_file = os.path.join(test_root, f)
        file_list.append(cur_file)
    pc_num_counter = 0
    pc_data = np.load(file_list[pc_num_counter])
    #pc_data = np.fromfile(file_list[pc_num_counter], dtype=np.float32).reshape(-1, 4)
    print(pc_data)
    print(pc_data.shape)

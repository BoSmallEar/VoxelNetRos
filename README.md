## Voxelnet ROS Implementation 
----
VoxelNet Implementation codes from "https://github.com/qianguih/voxelnet"

### Dependencies
* python3.5+
* TensorFlow (tested on 1.4)
* opencv
* shapely
* numba
* easydict
* ROS(melodic)
* jsk package(ros-jsk-recognition)
* cv_brdge(ros)
* nodelet(ros)
* pcl_ros(ros)
* rviz(ros)
  
### Data Preparation
Download the test data: https://pan.baidu.com/s/1kxZxrjGHDmTt-9QRMd_kOA

unzip to `data` folder,Directory structure should be:

```
data
----npy_lidar_0048(lidar point cloud)
----result_road_0048(road detection result)
----rgb_image_0048(correponding rgb left camera image)
```
### Instructions
before run the code, you may need to install:


- clone this repository
- move voxelnet_ros folder to your `catkin_ws`
- `catkin_make`
- `roscd voxelnet/script/`
- `you probably also need to complie the box_overlap.so using numba
- `python3 voxelnet_ros.py & python3 pub_kitti_point_cloud.py`& python3 pub_rgb_image.py & python3 pub_road_result.py 
  - unfortunately, `rosrun` won't work. because it's using Python 3.x instead of 2.x
- `roslaunch voxelnet_ros convex_hull.launch
- `rosrun rviz rviz 
  - use the voxelnet.rviz to config rviz

### ROS Node 

<img src="./pictures/8.png" />

### Rviz Animation
<img src="./pictures/1.png" />
<img src="./pictures/2.png" />
<img src="./pictures/3.png" />



### Future Work
- Retrain the model
- Still too slow, need more efficient implementation for the VoxelNet
- code clean

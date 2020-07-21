
#!/bin/bash
sourFolder="./lidar_2d/"
targetFolder="./npy_lidar_0048/"
 
for file in `ls $sourFolder | grep 2011_09_26_0048` ;
do
	echo "----------processing file is $file-----------"
	cp $sourFolder$file $targetFolder
	echo "---------------------------------------------"
done

 

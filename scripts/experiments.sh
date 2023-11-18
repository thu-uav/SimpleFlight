#!/bin/bash

# A simple Bash script with a for loop
filename="log_files_$(date).txt"
echo > $filename
ball_speeds="3.0 4.0 5.0 6.0 7.0"
drone_speeds="2.0 3.0 4.0 5.0"
for ball_speed in $ball_speeds
do
    for drone_speed in $drone_speeds
    do
        echo "Current time: $(date)" >> $filename
        echo "ball speed = $ball_speed, drone speed = $drone_speed" >> $filename
        $ISAACSIM_PATH/python.sh scripts/train.py task.ball_speed=$ball_speed task.target_vel=[0.0,$drone_speed,0.0]
    done
done
先搭建基础的实验环境

[Reinforced Imitation](<https://github.com/ethz-asl/rl-navigation>)

因为含有一些权重数据，这个repo比较大(已经放在了rec)

```bash
git clone https://github.com/ethz-asl/rl-navigation
```

同时需要一个simulator

```bash
git clone https://github.com/ros-simulation/stage_ros/ -b add_pose_sub
```

也需要turtlebot bringup

请使用16.04的对应源，修改`/etc/apt/sources.list`

```bash
sudo apt install ros-kinetic-turtlebot
sudo apt install ros-kinetic-turtlebot-simulator
```

```bash
export ROS_HOSTNAME=localhost
```





#### Training the Model

1. First run the stage simulator: `roslaunch reinforcement_learning_navigation stage_sim.launch`
2. In a separate terminal, run the training code: `rosrun reinforcement_learning_navigation train_cpo.py --output_name $experiment_name$`
   In order to use pre-trained weights from imitation learning, add the arguments `--jump_start 1 --model_init $path_to_policy_weights$`
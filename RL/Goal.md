### Original Ideas

#### Problem

Task-Based Navigation and Obstacles Avoidance. ~~not pure Navigation~~

<img src="./Goal.assets/goal_define.png" style="zoom:100%">

#### Features

- Map-less

- Autonomous navigation and task completion

- Task definition by imitating

#### Approach

***combine one or two new techniques***

1. **Data Generating ** 

   Use classic methods(SLAM, ROS move_base) to generate data in static env(without dynamic obstacles) [with random settings].

   - Dynamic obstacles will be handled by RL
   - Random settings: Avoid overfitting to some specific env 
     - Random static wall layout

2. **Task Definition by Imitating** 

   Imitation Learning(IL) to pre-train SRL(State Representation Learning) and RL(Constrained Policy Optimization (CPO)) 

   - Imitation: task define & accelerate training process by reduce random exploration iterations
   - SRL: reduce search space
   - CPO: avoid collision
   - NN Structure: 1D CNN

3. **Training Process ** 

   Train SRL and RL in dynamic env [with random settings] by trial and error exploration combined with reward signals. 

   - Dynamic obstacle avoidance with RL
   - Random settings: Avoid overfitting to some specific env 
     - Random static wall layout
     - Random dynamic obstacles

4. **Optimal Structure Search: **

   - AutoRL
     - automates the search for RL reward and neural network architecture

#### Contributions

1. Combine IL, SRL, RL together in task-based navigation
2. Pre-Training in static env and Re-Training in dynamic env
3. Use AutoRL to define rewards in staged-goals

**Other ideas**

Inverse RL 

hierarchical RL

Navigation: A* algorithms

#### Experiments

Multiple Approaches Comparation

- with/without Imitation Learning
- with/without SRL
- pure RL

#### Ref Papers

- [github RL navigation](<https://github.com/ethz-asl/rl-navigation>)

  [arxiv1805: Reinforced Imitation: Sample Efficient Deep Reinforcement Learning for Map-less Navigation by Leveraging Prior Demonstrations](<https://arxiv.org/abs/1805.07095>)

- [S-RL Toolbox’s documentation](<https://s-rl-toolbox.readthedocs.io/en/latest/>)

- [Google AI Navigation via AutoRL](<https://ai.googleblog.com/2019/02/long-range-robotic-navigation-via.html>)

#### Others

- [Lab for Autonomous and Intelligent Robotics ](<https://www.lair.hmc.edu/>)


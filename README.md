#### 文章目录

1. 机器人抓取与操作简介
  1. 抓取问题描述
  2. 抓取检测方法学
        1. 分析法
            2. 数据驱动（经验）法
                3. 分析法 vs. 数据驱动法
  3. 数据驱动法的分类
        1. 抓取已知的对象
            2. 抓取相似的对象
                3. 抓取未知的对象
                    4. 抓取Pipeline
2. 移动抓取系统组成
3. 技术点
  1. 机械臂与标定（相机配置方案）
        1. Eye-in-hand
            2. Eye-to-hand
  2. 物体识别与姿态估计
  3. 抓取检测
        1. 有已知定位和姿势
            2. 具有已知定位和无姿态的方法
                3. 无定位无姿态的方法
                    4. 抓取检测库GPD
  4. 机械臂运动规划
        1. 已有抓取点
            2. 无抓取点
  5. 抓取末端执行器
4. 附A: 机器人抓取硬件配置
  1. 抓取未知物体类（深度学习方法）
5. 机器人抓取开源项目
  1. 机器人抓取开源项目

## 1. 机器人抓取与操作简介

### 1.1 抓取问题描述

抓取问题可描述为以下三个问题：

1. 

抓取规划（Grasp Planning）

给定一个物体，给定一个任务，给定一个手，怎么去抓这个物体才是最好的？

在抓取规划中有两个比较重要的概念：

形封闭（Form closure）：如果一组静止约束能阻止物体所有的运动，则可实现物体的形封闭，如果这些约束是由机器人手指提供的，则称之为形封闭抓握（form-closure grasp）。

力封闭（Force closure）：可以理解为带摩擦的形封闭。考虑单个可移动物体和多个摩擦接触，如果一个物体因一组施加在物体表面的静态约束而不能发生任何位姿的改变，而这组静态约束完全由于机器人的手指施加在物体接触点处的力旋量决定，那么就成这种状态为力封闭状态，同时称这组抓取为力封闭抓取。

1. 

抓取控制（Grasp Control）

抓取控制是指通过与手接触来约束物体动力学，从而控制物体运动的问题。这个问题就是研究力控，包括手指末端的力控，触觉控制等，刚度控制，阻抗控制等等。很长一段时间，大家都在试图计算什么样的手指抓取力才是最优的，这里面以Martin Buss和李泽湘老师组的工作最为著名，将一个非线性优化问题转化成一个线性矩阵不等式问题，基本在几十毫秒左右可以得到优化结果。最近的这个方面的最好的工作应该算是DLR出来的 object-level impedance control（IJRR）。

1. 

灵巧操作（Dexterous Manipulation）

灵巧操纵物体。

### 1.2 抓取检测方法学

#### 1.2.1 分析法

分析法是指使用多指机械手构造**力封闭抓取** 的方法，该机械手灵巧、平衡、稳定，并表现出一定的动态行为。然后，抓取通常被描述为一个约束优化问题，其标准是衡量这四个属性中的一个或多个属性。

#### 1.2.2 数据驱动（经验）法

基于数据驱动的方法依赖于对对象的候选抓取样本，并根据特定的度量来对它们进行排序。该过程通常基于一些现有的抓取经验，这些经验可以是启发式的，也可以是在仿真或真实机器人上生成的。如何采样抓取候选集，抓取质量如何被评估以及好的抓取如何被表示，不同数据驱动方法在这些方面都有所不同。

#### 1.2.3 分析法 vs. 数据驱动法

与分析方法相反，遵循数据驱动范式的方法更加重视对象表示和感知处理，例如特征提取、相似性度量、对象识别或分类以及姿势估计。然后使用得到的数据从知识库或样本中检索抓取，并与现有的抓取经验进行比较。抓握的参数化不那么具体(例如，用接近向量代替指尖位置)，因此可以适应感知和执行中的不确定性。数据驱动方法不能保证上述灵巧性、平衡性、稳定性和动态行为的标准。它们只能通过经验加以验证。然而，它们构成了研究掌握动力学和进一步发展更接近现实的分析模型的基础。

### 1.3 数据驱动法的分类

基于数据驱动的方法可以按照机器人对抓取对象的了解程度来分类。**目前比较成熟的还是抓取已知三维信息的物体。**

#### 1.3.1 抓取已知的对象

这类方法假设以前遇到过抓取对象，并且已经为它生成了抓取。通常，机器人可以访问包含几何对象模型的数据库，这些模型与一些良好的抓取相关联。这个经验数据库通常是离线建立的。一旦物体被识别出来，目标就是**估计它的姿态** 并获得合适的抓取。
![在这里插入图片描述](https://cdl.itadn.com/b/weblog/blog-img/images/wklEMHoq58ZOFmntbf9XeCv34JLp.png)

#### 1.3.2 抓取相似的对象

很多情况下，抓取的目标对象与现有数据库的模型并不完全相同，但是在模型库中相似的同一类的物体，这便涉及到对相似物体的抓取。在目标对象被定位以后，利用基于关键点对应算法便可以将抓取点从模型库中存在的相似三维模型上转移到当前的局部对象中。由于当前的目标对象与数据库中的对象不完全相同，所以这类型的抓取算法是不需要进行六维姿态估计的。[Andrew等人](Miller A T, Knoop S, Christensen H I, et al. Automatic grasp planning using shape primitives[C]//2003 IEEE International Conference on Robotics and Automation (Cat. No. 03CH37422). IEEE, 2003, 2: 1824-1829. )提出了一种基于分类法的方法，该方法将对象划分为各个类别，每个类别均存在对应的抓取规范。[Tian等人](Tian H, Wang C, Manocha D, et al. Transferring Grasp Configurations using Active Learning and Local Replanning[C]//2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019: 1622-1628. )提出了一种将抓取构型从先前的示例对象转移到新目标上的方法，该方法假设新对象和示例对象具有相同的拓扑结构和相似的形状。他们考虑几何形状和语义形状特征对对象进行三维分割，利用主动学习算法为示例对象的每个部分计算一个抓取空间，并为新对象在模型部分和相应的抓取之间建立双射接触映射。这一类型的方法依赖于目标分割的准确性。然而，训练一个能识别出广泛对象的网络并不容易。同时，这些方法要求待抓取的三维物体与标注模型相似，以便找到相应的抓取模型。在经常发生遮挡的杂乱环境中，计算高质量的物体抓取点也是一个挑战。
![在这里插入图片描述](https://cdl.itadn.com/b/weblog/blog-img/images/70P5ygc92CtlDFjSexv1dETmKh3J.png)

#### 1.3.3 抓取未知的对象

该类方法并不假定具有对象的模型或任何形式的抓取经验。它们专注于识别对象结构或特征的感知数据，来生成和排名**候选抓取** 。它们通常是基于传感器感知到的物体的局部或全局特征。
![在这里插入图片描述](https://cdl.itadn.com/b/weblog/blog-img/images/4HYM5Rr03P98cJEpLKw7VgQUSCGW.png)

### 1.4 抓取Pipeline

- 场景分割：识别人类意图，对点云进行分割
- 目标表征和识别：理解对象 —— 目标检测，表征（点云、多面体、超二次曲面和高斯过程隐式曲面）
- 抓取规划：抓取数据库，力封闭，抓取质量
- 抓取执行：开环控制、力控制还是阻抗控制
  ![在这里插入图片描述](https://cdl.itadn.com/b/weblog/blog-img/images/tUzDPM2hVXnabmOT0BQ69I1jfCSJ.png)

## 2. 移动抓取系统组成

- 自主移动机器人
- 机械臂和标定
- 物体识别与姿态估计
- 抓取检测
- 运动规划
- 末端执行器

## 3. 技术点1: 机械臂与标定（相机配置方案）

机器人和摄像机的配置可分为两类构型：

```
1. Eye-to-hand：摄像机固定，与机器人基坐标系相对位置不变
2. Eye-in-hand：摄像机安装在机器人末端，随机器人一起移动
```

两种构型均需要进行**手眼标定** ，也即求解相机坐标系相对机器人坐标系的关系，两者均可统一到：

AX=XB*A**X*=*XB*

 进行求解，求解算法Tsai参考：https://github.com/realjc/handeye-calibration

- ROS也提供手眼标定功能包：visp_hand2eye_calibration 和 easy_handeye （https://github.com/IFL-CAMP/easy_handeye）
- Moveit也有手眼标定包：https://ros-planning.github.io/moveit_tutorials/doc/hand_eye_calibration/hand_eye_calibration_tutorial.html

#### 3.1 Eye-in-hand

![在这里插入图片描述](https://cdl.itadn.com/b/weblog/blog-img/images/k0O6p8vFfmrXQGgjABbDZnLtPKW5.png)



#### 3.2 Eye-to-hand

![在这里插入图片描述](https://cdl.itadn.com/b/weblog/blog-img/images/Hl7h3RvkJ8S5c2CKm1rqdwnVzaFB.png)



## 4. 技术点2: 物体识别与姿态估计

对于**已知的物体** （1.3.1节）的抓取，可以通过学习已有的成功抓取实例，再结合具体环境进行机器人抓取。事实上，如果目标对象已知，则意味着对象的三维模型和抓取点位置在数据库中也是先验已知的。这种情况下，只需要从局部视图估计目标对象的**6D位姿** ，并通过ICP等算法进行姿态细化与精确微调，进一步便可以得到目标物体的抓取位置。这是目前已知的抓取系统中最流行的方法，也是在亚马逊抓取挑战赛中普遍使用的算法。

**6D姿态估计方法汇总**
6D姿态估计帮助机器人知道要抓取的物体的位置和方向。姿态估计方法大致可分为四种，分别基于**对应、模板、投票和回归** 。

不基于目标检测先验信息

**基于对应方法：**

在2D点和3D点之间找到匹配，并使用PNP方法,如SIFT、SURF、ORB
通过随机假设或三维描述寻找三维对应关系，并使用ICP对结果进行细化： FPFH、SHOT
**基于模板方法：**LineMod算法。从没有纹理的三维模型渲染图像，提取梯度信息进行匹配，并使用ICP对结果进行细化。
**基于投票方法：**PPF算法。基于三维点云或具有姿态的渲染RGB-D图像，每个局部预测一个结果，并使用RANSAC对结果进行优化。

基于**目标检测** 信息的6D姿态估计

这种方法也称为基于回归的方法，它同时完成目标检测和6D姿态估计。基于回归的方法：BB8、SD6D、PoseCNN、Deep6DPose。基于三维点云或具有适当姿势的渲染RGB-D图像，并使用CNN进行姿态回归。

独立于目标检测的方法在通常发生遮挡的杂乱场景中显示出明显的局限性；基于目标检测的方法是缺乏足够的训练数据。

基于目标检测的方法示例：

物体识别和6D姿态估计**训练数据** 准备：

```
 * 如果物体有CAD模型的话，可以直接使用模型进行渲染，获得物体数据集
 * 如果没有CAD模型，则需要使用**三维重建平台** ，**扫描** 物体获得较为准确的三维模型
 * 然后通过截取模型的各项视图，叠加随机背景，实现物体数据集的自动生成（类别及姿态）
```

分割点云
![在这里插入图片描述](https://cdl.itadn.com/b/weblog/blog-img/images/W6BNdhgTqHbuA7rMC8XxFG5EYlwV.png)

匹配和姿态估计
![在这里插入图片描述](https://cdl.itadn.com/b/weblog/blog-img/images/4VCT7HarqL1nY5Np3RS0floKxzIc.png)

## 5. 技术点3: 抓取检测

抓取检测被定义为能够识别任何给定图像中物体的抓取点或抓取姿势。抓取策略应确保对新物体的稳定性、任务兼容性和适应性，抓取质量可通过物体上接触点的位置和手的配置来测量。为了掌握一个新的对象，完成以下任务，有分析方法和经验方法。分析方法根据抓取稳定性或任务要求的运动学和动力学公式，选择手指位置和手部构型，经验方法根据具体任务和目标物体的几何结构，使用学习算法选择抓取。根据是否需要进行目标定位，需要确定目标的姿态，进一步将其分为三类：具有已知定位和姿势的方法、具有已知定位和无姿态的方法、无定位和无姿态的方法。

### 5.1 具有已知定位和姿势

针对已知目标的经验方法，利用姿态将已知目标的抓取点转换为局部数据。主要算法有：

- Multi-view self-supervised deep learning for 6d pose estimation in the amazon picking challenge.
- Silhonet: An RGB method for 3d object pose estimation and grasp planning.

### 5.2 具有已知定位和无姿态的方法

主要方法：

- Automatic grasp planning using shape primitives.
- Part-based grasp planning for familiar objects.
- Transferring grasp configurations using active learning and local replanning.
- Dex-net 2.0: Deep learning to plan robust grasps with synthetic point clouds and analytic grasp metrics.

### 5.3 无定位无姿态的方法

主要基于深度学习方法，包括：

- Deep learning for detecting robotic grasps.
- Real-time grasp detection using convolutional neural networks.
- Object discovery and grasp detection with a shared convolutional neural network.
- Supersizing self-supervision: Learning to grasp from 50k tries and 700 robot hours.
- Real-time, highly accurate robotic grasp detection using fully convolutional neural networks with high-resolution images.
- Robotic pick-and-place of novel objects in clutter with multi-affordance grasping and cross-domain image matching.

### 5.4 抓取检测库GPD

- 地址：https://github.com/atenpas/gpd
- **抓取姿态检测 (Grasp Pose Detection, GPD)** 用于检测三维点云中**两指机器人手**(如平行下颌夹持器)的六自由度抓取姿态(3自由度位置和3自由度方向)。GPD将一个点云作为输入，并生成可以握持的姿态估计值作为输出。
- GPD的主要优势是:
  - 适用于未知对象(检测不需要CAD模型)
  - 可在杂乱物体中工作，并输出6自由度抓取姿势

## 6. 技术点4: 机械臂运动规划

### 6.1 已有抓取点

假设抓取点已检测到。这些方法设计了从机器人手到目标物体抓取点的路径。这里运动表示是关键问题。虽然存在从机器人手到目标抓握点的无限数量的轨迹，但是由于机器人臂的限制，许多区域无法到达。因此，需要对轨迹进行规划。主要有三种方法，如传统的基于DMP的方法、模仿学习的方法和基于强化学习的方法。

**基于DMP的方法：**主要包括DMP算法。形式化为稳定的非线性吸引子系统。

**基于模仿学习的方法：**Generalization of human grasping for multi-fingered robot hands.

**基于强化学习的方式：**Task-agnostic self-modeling machines.

- 一般而言，检测到抓取点后，运动规划常用ROS MoveIt组件实现
- ![在这里插入图片描述](https://cdl.itadn.com/b/weblog/blog-img/images/GOtybN4qxjMvAu2ZV9FmWlpIf6nP.png)

### 6.2 无抓取点

主要通过强化学习直接完成对原始RGB-D图像的抓取任务。

主要有：

- Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection.
- Qt-opt: Scalable deep reinforcement learning for vision-based robotic manipulation.
- Robustness via retrying: Closed-loop robotic manipulation with selfsupervised learning.

## 7. 技术点5——抓取末端执行器

**二指夹爪** 是研究和应用中非常常见的抓取末端执行器，例如5.4中的抓取姿态检测库就适用于二指夹爪

**吸盘** 也是比较常见的末端执行器

**灵巧手** 的设计也是机器人抓取的一个重要研究方向，例如软性灵巧手：既不会损坏物体，还具有一定刚度能实现物体抓取

## 8. 附A: 机器人抓取硬件配置

### 8.1 抓取未知物体类（深度学习方法）

实例1（Article：A New Approach Based on Two-stream CNNs for Novel Objects Grasping in Clutter）：

```
* 机械臂：UR5
* 深度相机：Asus Xtion Pro sensor
* CPU：2.0 GHz Intel Corei7-4510U
* 内存：32 GB
* 显卡：Nvidia GeForce 840M （**算力5.0** ）
* 平台：ROS + MoveIt!
```

实例2（An Efficient Robotic Grasping Pipeline Base on Fully Convolutional Neural Network）

```
* 机械臂：Kinova Mico 6-DOF robot
* 夹爪：Kinova KG-3 3-fingered gripper
* CPU：3.4 GHz Intel Core i7-6700 8-Core
* 显卡：NVIDIA GeForce GTX 1080-Ti （**算力6.1** ）
```

## 9. 附B: 机器人抓取开源项目

1. 

1.ros2grasp（包含手眼标定）

https://intel.github.io/ros2_grasp_library/docs/doc/overview.html
https://github.com/intel/ros2_grasp_library
![在这里插入图片描述](https://cdl.itadn.com/b/weblog/blog-img/images/1xzyWmHJ5O8Fd72Yr0jeCZXgn3up.png)

1. 

graspit

http://graspit-simulator.github.io/

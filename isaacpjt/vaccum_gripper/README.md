# isaac_pick_and_place

issac sim -> 6
```bash
export ROS_DOMAIN_ID=140
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/exts/isaacsim.ros2.bridge/humble/lib

isaac_python m0609_pick_place_dual_tray.py
```

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<다른_PC의_Isaac_Sim_ROS2_Bridge_lib_경로>

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=140
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

source ~/cobot3_ws/install/setup.bash
ros2 run cobot3 m0609_color_detector
```

# 다음 과제에서의 개선 필요 사항
이미지 보내는 과정에서 외부 pc에서 토픽 구독시 느려지는 현상 발생
1. python 코드로 시뮬레이션 내부의 카메라에 연결하여 이미지 받아오기
2. 이미지 전처리 -> crop or downscaling 
3. ros bridge를 이용해서 -> 토픽 발행
4. 전체 알고리즘 구조 확인해서 중복연산 및 대기시간 최소화, 최적화 필수


.bashrc
```bash
source /opt/ros/humble/setup.bash


alias isaac='~/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/isaac-sim.sh'

alias isaac_python="~/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh"
```
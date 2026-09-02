from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})     # 1. Application 시물레이터 실행

import numpy as np
import time
import omni.usd
from isaacsim.core.api import World # isaacsim에서 api 로드
from isaacsim.core.api.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)                # 2. World 월드 기본 단위 정의
stage = omni.usd.get_context().get_stage()              # 3. Stage USD 씬 데이터에 직접 접근

cube_red = DynamicCuboid(                              # 4. Prim issacsim에서 사용하는 prim 정의
    prim_path="/World/RedCube",
    name="red_cube",
    position=np.array([2.0, -2.0, 5.0]),
    scale=np.array([0.15, 0.15, 0.15]),
    color=np.array([1.0, 0.0, 0.0]),
)
cube_green = DynamicCuboid(                              # 4. Prim issacsim에서 사용하는 prim 정의
    prim_path="/World/GreenCube",
    name="green_cube",
    position=np.array([0.0, 0.0, 5.0]),
    scale=np.array([0.15, 0.15, 0.15]),
    color=np.array([0.0, 1.0, 0.0]),
)
cube_blue = DynamicCuboid(                              # 4. Prim issacsim에서 사용하는 prim 정의
    prim_path="/World/BlueCube",
    name="blue_cube",
    position=np.array([-2.0, 2.0, 5.0]),
    scale=np.array([0.15, 0.15, 0.15]),
    color=np.array([0.0, 0.0, 1.0]),
)

world.scene.add_default_ground_plane()                  # 5. Scene 씬에 평면 추가
world.scene.add(cube_red) # 사용자 정의 prim 추가
world.scene.add(cube_green) # 사용자 정의 prim 추가
world.scene.add(cube_blue) # 사용자 정의 prim 추가

world.reset() # 물리 초기화 및 씬 확정

while simulation_app.is_running():                      # 6. Simulation
    world.step(render=True)

simulation_app.close()
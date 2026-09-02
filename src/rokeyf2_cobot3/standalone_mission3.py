from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})     # 1. Application 시물레이터 실행

import numpy as np
import time
import omni.usd
from isaacsim.core.api import World # isaacsim에서 api 로드
from isaacsim.core.api.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)                # 2. World 월드 기본 단위 정의
stage = omni.usd.get_context().get_stage()              # 3. Stage USD 씬 데이터에 직접 접근

cube_big = DynamicCuboid(                              # 4. Prim issacsim에서 사용하는 prim 정의
    prim_path="/World/BigCube",
    name="big_cube",
    position=np.array([0.0, 0.0, 0.15]),
    scale=np.array([0.3, 0.3, 0.3]),
    color=np.array([1.0, 0.0, 0.0]),
)
cube_small = DynamicCuboid(                              # 4. Prim issacsim에서 사용하는 prim 정의
    prim_path="/World/SmallCube",
    name="small_cube",
    position=np.array([0.0, 0.0, 5.0]),
    scale=np.array([0.1, 0.1, 0.1]),
    color=np.array([0.0, 1.0, 0.0]),
)

world.scene.add_default_ground_plane()                  # 5. Scene 씬에 평면 추가
world.scene.add(cube_big) # 사용자 정의 prim 추가
world.scene.add(cube_small) # 사용자 정의 prim 추가

world.reset() # 물리 초기화 및 씬 확정

while simulation_app.is_running():                      # 6. Simulation
    world.step(render=True)

simulation_app.close()
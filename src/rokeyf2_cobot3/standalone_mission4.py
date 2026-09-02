from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})     # 1. Application 시물레이터 실행

import numpy as np
import time
import omni.usd
from isaacsim.core.api import World # isaacsim에서 api 로드
from isaacsim.core.api.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)                # 2. World 월드 기본 단위 정의
stage = omni.usd.get_context().get_stage()              # 3. Stage USD 씬 데이터에 직접 접근

world.scene.add_default_ground_plane()                  # 5. Scene 씬에 평면 추가
world.reset() # 물리 초기화 및 씬 확정

step_count = 0

while simulation_app.is_running():                      # 6. Simulation
    world.step(render=True)
    time.sleep(0.01)
    step_count += 1

    if step_count % 100 == 0:
        print(f"step: {step_count}")

simulation_app.close()
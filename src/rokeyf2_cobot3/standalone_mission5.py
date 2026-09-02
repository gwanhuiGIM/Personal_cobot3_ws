from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})     # 1. Application 시물레이터 실행

import numpy as np
import time
import omni.usd
from isaacsim.core.api import World # isaacsim에서 api 로드
from isaacsim.core.api.objects import DynamicCuboid
import omni.timeline

world = World(stage_units_in_meters=1.0)                # 2. World 월드 기본 단위 정의
stage = omni.usd.get_context().get_stage()              # 3. Stage USD 씬 데이터에 직접 접근

cube_prim = DynamicCuboid(                              # 4. Prim issacsim에서 사용하는 prim 정의
    prim_path="/World/RedCube",
    name="red_cube",
    position=np.array([0.0, 0.0, 0.15]),
    scale=np.array([0.3, 0.3, 0.3]),
    color=np.array([1.0, 0.0, 0.0]),
)

world.scene.add_default_ground_plane()                  # 5. Scene 씬에 평면 추가
world.scene.add(cube_prim)
world.reset() # 물리 초기화 및 씬 확정

step_count = 0

timeline = omni.timeline.get_timeline_interface()

def on_event(event):
    if event.type == int(omni.timeline.TimelineEventType.PLAY):
        global step_count
        step_count = 0
        print("[리셋] Play 시작 -> step_count = 0")

subscription = timeline.get_timeline_event_stream().create_subscription_to_pop(
    on_event
)

while simulation_app.is_running():                      # 6. Simulation
    world.step(render=True)
    time.sleep(0.01)
    step_count += 1

    if step_count % 100 == 0:
        print(f"step: {step_count}")

    if step_count == 300 :
        print("[이동] 큐브 순간 이동")
        cube_prim.set_world_pose(
            position=np.array([0.0, 0.0, 5.0])
        )
    

simulation_app.close()

#하이

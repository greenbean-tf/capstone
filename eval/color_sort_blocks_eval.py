import math

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import MassPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.seed import configure_seed

from leisaac.utils.domain_randomization import domain_randomization, randomize_object_uniform
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from simulator import ASSETS_ROOT
from simulator.assets.scenes.living_room import LIVING_ROOM_CFG, LIVING_ROOM_USD_PATH
from simulator.tasks.template.single_arm_franka_cfg import (
    SingleArmFrankaObservationsCfg,
    SingleArmFrankaTaskEnvCfg,
    SingleArmFrankaTaskSceneCfg,
    SingleArmFrankaTerminationsCfg,
)

LIVING_OBJECTS_ROOT = ASSETS_ROOT / "scenes" / "living_room" / "objects"
OBJECT_Z: float = 0.05
BASKET_Z: float = 0.05

configure_seed(42)


@configclass
class ColorSortBlocksEvalSceneCfg(SingleArmFrankaTaskSceneCfg):
    scene: AssetBaseCfg = LIVING_ROOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    green_block: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/green_block",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVING_OBJECTS_ROOT / "Bridge" / "Bridge.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.32, -0.35, OBJECT_Z),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
    )

    blue_block: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/blue_block",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVING_OBJECTS_ROOT / "Cylinder" / "Cylinder.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, -0.35, OBJECT_Z),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
    )

    red_block: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/red_block",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVING_OBJECTS_ROOT / "Triangle" / "Triangle.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.58, -0.35, OBJECT_Z),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
    )

    # Green basket — right side
    green_basket: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/green_basket",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.65, -0.55, BASKET_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVING_OBJECTS_ROOT / "Storage_Box" / "storage_box.usd"),
            mass_props=MassPropertiesCfg(mass=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
        ),
    )

    # Blue basket — front/middle
    blue_basket: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/blue_basket",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, -0.55, BASKET_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVING_OBJECTS_ROOT / "Storage_Box" / "storage_box.usd"),
            mass_props=MassPropertiesCfg(mass=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.9)),
        ),
    )

    # Red basket — left side
    red_basket: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/red_basket",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.05, -0.55, BASKET_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVING_OBJECTS_ROOT / "Storage_Box" / "storage_box.usd"),
            mass_props=MassPropertiesCfg(mass=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
    )


def each_block_in_correct_basket(
    env,
    green_block_cfg: SceneEntityCfg,
    blue_block_cfg: SceneEntityCfg,
    red_block_cfg: SceneEntityCfg,
    green_basket_cfg: SceneEntityCfg,
    blue_basket_cfg: SceneEntityCfg,
    red_basket_cfg: SceneEntityCfg,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> torch.Tensor:
    """Success when every block is inside its colour-matched basket."""
    pairs = [
        (green_block_cfg.name, green_basket_cfg.name),
        (blue_block_cfg.name, blue_basket_cfg.name),
        (red_block_cfg.name, red_basket_cfg.name),
    ]
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for block_name, basket_name in pairs:
        block: RigidObject = env.scene[block_name]
        basket: RigidObject = env.scene[basket_name]
        block_pos = block.data.root_pos_w - env.scene.env_origins
        basket_pos = basket.data.root_pos_w - env.scene.env_origins
        done = torch.logical_and(done, block_pos[:, 0] < basket_pos[:, 0] + x_range[1])
        done = torch.logical_and(done, block_pos[:, 0] > basket_pos[:, 0] + x_range[0])
        done = torch.logical_and(done, block_pos[:, 1] < basket_pos[:, 1] + y_range[1])
        done = torch.logical_and(done, block_pos[:, 1] > basket_pos[:, 1] + y_range[0])
        done = torch.logical_and(done, block_pos[:, 2] < basket_pos[:, 2] + z_range[1])
        done = torch.logical_and(done, block_pos[:, 2] > basket_pos[:, 2] + z_range[0])
    return done


@configclass
class EvalTerminationsCfg(SingleArmFrankaTerminationsCfg):
    success = DoneTerm(
        func=each_block_in_correct_basket,
        params={
            "green_block_cfg": SceneEntityCfg("green_block"),
            "blue_block_cfg": SceneEntityCfg("blue_block"),
            "red_block_cfg": SceneEntityCfg("red_block"),
            "green_basket_cfg": SceneEntityCfg("green_basket"),
            "blue_basket_cfg": SceneEntityCfg("blue_basket"),
            "red_basket_cfg": SceneEntityCfg("red_basket"),
            "x_range": (-0.12, 0.12),
            "y_range": (-0.12, 0.12),
            "z_range": (-0.08, 0.08),
        },
    )


@configclass
class ColorSortBlocksEvalEnvCfg(SingleArmFrankaTaskEnvCfg):
    scene: ColorSortBlocksEvalSceneCfg = ColorSortBlocksEvalSceneCfg(env_spacing=8.0)
    observations: SingleArmFrankaObservationsCfg = SingleArmFrankaObservationsCfg()
    terminations: EvalTerminationsCfg = EvalTerminationsCfg()
    task_description: str = (
        "pick up each toy block and place it into the matching colour basket: "
        "green block into the green basket, blue block into the blue basket, "
        "red block into the red basket."
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (0.8, 0.87, 0.67)
        self.viewer.lookat = (0.4, -1.3, -0.2)
        self.dynamic_reset_gripper_effort_limit = False

        self.scene.robot.init_state.pos = (0.35, -0.74, 0.01)
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)
        self.scene.robot.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -math.pi / 4.0,
            "panda_joint3": 0.0,
            "panda_joint4": -3.0 * math.pi / 4.0,
            "panda_joint5": 0.0,
            "panda_joint6": math.pi / 2.0,
            "panda_joint7": math.pi / 4.0,
            "panda_finger_joint1": 0.04,
            "panda_finger_joint2": 0.04,
        }

        parse_usd_and_create_subassets(LIVING_ROOM_USD_PATH, self)

        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "green_block",
                    pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
                ),
                randomize_object_uniform(
                    "blue_block",
                    pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
                ),
                randomize_object_uniform(
                    "red_block",
                    pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
                ),
            ],
        )


TASK_ID = "Private-ColorSortBlocks-Eval-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:ColorSortBlocksEvalEnvCfg"},
)

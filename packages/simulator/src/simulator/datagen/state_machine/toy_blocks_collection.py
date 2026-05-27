"""State machine for the Franka toy-blocks-collection task."""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
)

from leisaac.datagen.state_machine.base import StateMachineBase

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_OBJECT_NAMES = ("green_block", "blue_block", "red_block")
_STORAGE_BOX_NAME = "storage_box"
_EE_BODY_NAME = "panda_hand"
_FRANKA_ARM_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)
_FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0

_MAX_CARTESIAN_DELTA = 0.018
_MAX_ROT_DELTA = 0.08
_IK_DLS_LAMBDA = 0.003

_HOVER_Z_OFFSET = 0.3
_GRASP_Z_OFFSET = 0.08
_LIFT_Z_OFFSET = 0.3
_RELEASE_Z_OFFSET = 0.09
_GRIPPER_DOWN_ROLL_W = math.pi
_GRIPPER_DOWN_PITCH_W = 0.0
_GRIPPER_DOWN_YAW_OFFSET_RANGE = (-0.15, 0.15)
# Grasp yaw bias (rad) on top of the object's world yaw, before the random
# jitter. π/2 because the gripper's fingers open along the EE local Y axis,
# 90° from the detected object +x heading. Per-USD orientation correction
# lives in env_cfg's ``per_object_yaw_offset``.
_GRASP_YAW_OFFSET: float = math.pi / 2.0
# Horizontal retreat (m) toward the robot base applied to approach + grasp
# targets. Stops the EE from overshooting the object when the IK budget is tight
# at the edge of the workspace. Per-object because thin / hollow USDs (Bridge,
# Triangle) can miss the grip if the fingers land off-centre.
_GRASP_RETREAT_PER_OBJECT: dict[str, float] = {
    "green_block": 0.0,
    "blue_block": 0.025,
    "red_block": 0.0,
}
# Per-object z target during the grasp (close) phase, relative to obj_pos_w.
# Setting it below ``_GRASP_Z_OFFSET`` lets the EE descend further while the
# fingers close (helps short / hollow shapes like Bridge); equal to it keeps
# the EE still while fingers close (helps wide cylinders like the Cylinder USD).
_GRASP_Z_AT_CLOSE_PER_OBJECT: dict[str, float] = {
    "green_block": 0.02,
    "blue_block":  0.0,
    "red_block":   0.0,
}
# Per-object world-frame xy nudge added to the grasp anchor. Use this when a
# specific USD's centre-of-mesh is offset from the tag-detected pose.
_GRASP_XY_OFFSET_PER_OBJECT: dict[str, tuple[float, float]] = {
    "green_block": (0.0, 0.0),
    "blue_block": (-0.040, 0.0),
    "red_block": (0.0, 0.0),
}
# Per-object world-frame xy offset added to the storage box position when
# placing each object. Triangle layout so the three blocks land in clearly
# separated quadrants of the box rather than along the same line.
_DROP_XY_OFFSET_PER_OBJECT: dict[str, tuple[float, float]] = {
    "green_block": (-0.05, 0.04),
    "blue_block": (0.0, -0.05),
    "red_block": (0.05, 0.04),
}

_SUCCESS_X_RANGE = (-0.12, 0.12)
_SUCCESS_Y_RANGE = (-0.12, 0.12)
_SUCCESS_Z_RANGE = (-0.08, 0.08)

_FRANKA_REST_JOINT_POS = {
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

# Per-object phase durations: hover, approach, grasp, lift, move_above_box, lower, release/retreat
# Reference (cup_stacking.py): (160, 80, 20, 100, 85, 35, 30) for a single-object task.
# Toy-blocks adjustments vs the reference:
#   approach: 200 steps — extended for IK convergence near robot base singularity.
#   grasp: 40 steps — extra 20 steps so EE can descend fully to block before fingers close.
#   hover/lift: +20 steps each — hover/lift offsets are 0.30 m here vs 0.15–0.20 m in cup-stacking.
#   move_above_box: +25 steps — storage box can be up to ~0.4 m lateral travel from any block.
#   lower: matched to reference (35) — 15 was too short for a clean gripper release.
_PHASE_DURATIONS_PER_OBJECT = (200, 200, 40, 110, 200, 35, 30)
_PHASES_PER_OBJECT = len(_PHASE_DURATIONS_PER_OBJECT)

# ---------------------------------------------------------------------------
# Phase transition logic
# ---------------------------------------------------------------------------

# Minimum step count before early exit is checked, indexed by phase-in-cycle.
_PHASE_MIN_STEPS: tuple[int, ...] = (200, 40, 10, 30, 60, 35, 15)
#                                     ^    ^   ^   ^   ^   ^   ^
#                              hover  apr grsp lft mab lwr ret

# Phases that always run to full duration (no early exit):
#   0 — hover: linear-interpolated target reaches final position only at the last step.
#   5 — lower: block must settle in box before gripper opens.
# Phase 4 (move_above_box) uses convergence check so the robot waits until the
# EE is truly above the box before lowering, regardless of travel distance.
_FIXED_DURATION_PHASES: frozenset[int] = frozenset({0, 5})

# Phase 1 (approach): advance when EE is within this distance of the block
# centre (obj_pos_w).  Using the actual block position rather than the
# computed approach target avoids accumulated error from retreat/xy offsets.
# At a correct approach position (block + 8 cm Z, ~0–2.5 cm XY retreat) the
# 3-D distance to the block centre is ~8.4 cm, so 10 cm gives a safe margin.
# Do NOT tighten below ~0.10: the DLS IK oscillates near the exact target,
# and a stricter threshold forces the arm to stay in that noisy zone longer.
_EE_TO_BLOCK_THRESHOLD: float = 0.10  # 10 cm  (3-D)

# XY-only alignment check applied on top of the 3-D threshold.
# Ensures the gripper is laterally centred over the block before closing,
# independent of height.  Per-object because the cylinder requires tighter
# lateral alignment than upright blocks.
_APPROACH_XY_THRESHOLD_PER_OBJECT: dict[str, float] = {
    "green_block": 0.05,   # 5 cm — upright block, generous
    "blue_block":  0.03,   # 3 cm — cylinder on side, tight
    "red_block":   0.05,   # 5 cm — upright block, generous
}

# Phase 2 (grasp): advance when total finger width exceeds this (fingers did not
# close fully → something is between them).
# Fingers range 0.0 (closed) to 0.04 m each; sum > 0.008 m means object grasped.
_MIN_GRASP_WIDTH: float = 0.008  # metres, sum of both finger joint positions

# Phases 1 and 2: number of *consecutive* steps the condition must hold before
# advancing. Prevents false-positive exits caused by IK oscillation near
# singularities (EE briefly within threshold, then drifts away again).
_CONVERGENCE_HOLD_STEPS: int = 5

# Per-object override for phase 1 (approach) hold steps.
# The blue cylinder lying on its side has a smaller contact area than upright
# blocks, so the gripper XY position must be more stable before closing.
# More hold steps means the IK must settle for longer before transitioning,
# reducing premature exits while the arm is still oscillating.
_APPROACH_HOLD_STEPS_PER_OBJECT: dict[str, int] = {
    "green_block": _CONVERGENCE_HOLD_STEPS,
    "blue_block": 20,
    "red_block": _CONVERGENCE_HOLD_STEPS,
}

# Phase 3 (lift): advance when object z exceeds this height (successfully lifted).
_LIFT_SUCCESS_Z: float = 0.25   # metres above world origin

# Phase 6 (retreat): advance when EE is within this distance of retreat target.
_EE_CONVERGENCE_THRESHOLD: float = 0.025  # 2.5 cm

# ---------------------------------------------------------------------------
# Grasp failure detection (lift phase)
# ---------------------------------------------------------------------------
# If object z has not exceeded _MIN_LIFT_Z by step _GRASP_CHECK_STEP,
# the episode is aborted immediately.
_MIN_LIFT_Z: float = 0.10       # metres (block starts at OBJECT_Z ≈ 0.05)
_GRASP_CHECK_STEP: int = 80     # step within lift phase at which to check


def _constant_gripper(num_envs: int, device: torch.device, value: float) -> torch.Tensor:
    return torch.full((num_envs, 1), value, device=device)


def _clamp_delta(delta: torch.Tensor, max_norm: float = _MAX_CARTESIAN_DELTA) -> torch.Tensor:
    norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return delta * scale


def _shortest_quat(quat: torch.Tensor) -> torch.Tensor:
    return torch.where(quat[:, 0:1] < 0.0, -quat, quat)


def _retreat_xy_toward(
    target_pos_w: torch.Tensor,
    anchor_pos_w: torch.Tensor,
    distance: float,
) -> torch.Tensor:
    """Pull ``target_pos_w`` xy toward ``anchor_pos_w`` by ``distance`` metres."""
    out = target_pos_w.clone()
    delta_xy = out[:, :2] - anchor_pos_w[:, :2]
    norm = torch.linalg.norm(delta_xy, dim=-1, keepdim=True).clamp_min(1e-6)
    out[:, :2] -= distance * (delta_xy / norm)
    return out


def _yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Yaw (rotation about world z) from a (w, x, y, z) quaternion."""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _find_body_index(robot, body_name: str) -> int:
    if hasattr(robot, "find_bodies"):
        body_ids, _ = robot.find_bodies(body_name)
        if len(body_ids) > 0:
            return int(body_ids[0])

    body_names = getattr(robot.data, "body_names", None)
    if body_names is not None and body_name in body_names:
        return body_names.index(body_name)

    return -1


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class ToyBlocksCollectionStateMachine(StateMachineBase):
    """Scripted Franka policy for collecting toy blocks into a storage box.

    Picks up green_block, blue_block, and red_block sequentially and places
    each into the storage_box.  Each object goes through 7 phases:

    0. Move above object
    1. Approach down to object
    2. Close gripper to grasp
    3. Lift object upward
    4. Move above storage box
    5. Lower and release
    6. Retreat upward

    The action vector is ``[panda_joint1, ..., panda_joint7, gripper]``.
    """

    MAX_STEPS: int = len(_OBJECT_NAMES) * sum(_PHASE_DURATIONS_PER_OBJECT) + 100

    def __init__(self) -> None:
        self._step_count: int = 0
        self._episode_done: bool = False
        self._ee_body_idx: int = -1
        self._jacobi_body_idx: int = -1
        self._arm_joint_ids: list[int] = []
        self._jacobi_joint_ids: list[int] = []
        self._rest_joint_pos: torch.Tensor | None = None
        self._rest_ee_pos_w: torch.Tensor | None = None
        self._initial_ee_pos_w: torch.Tensor | None = None
        self._gripper_down_yaw_w: torch.Tensor | None = None
        self._gripper_down_yaw_offset_w: torch.Tensor | None = None
        self._current_object_idx: int = 0
        self._event: int = 0
        self._events_dt: list[int] = list(_PHASE_DURATIONS_PER_OBJECT) * len(_OBJECT_NAMES)
        self._finger_joint_ids: list[int] = []
        self._phase_convergence_count: int = 0
        # Cached from the last get_action() call for convergence checking in advance().
        self._last_target_pos_w: torch.Tensor | None = None
        self._last_ee_pos_w: torch.Tensor | None = None
        self._last_obj_pos_w: torch.Tensor | None = None
        self._last_finger_pos: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # StateMachineBase interface
    # ------------------------------------------------------------------

    def setup(self, env) -> None:
        robot = env.scene["robot"]
        self._ee_body_idx = _find_body_index(robot, _EE_BODY_NAME)
        joint_names = list(robot.data.joint_names)
        missing = [j for j in _FRANKA_ARM_JOINT_NAMES if j not in joint_names]
        if missing:
            raise ValueError(f"Missing Franka joints {missing} in {joint_names}")
        self._arm_joint_ids = [joint_names.index(j) for j in _FRANKA_ARM_JOINT_NAMES]
        self._finger_joint_ids = [joint_names.index(j) for j in _FINGER_JOINT_NAMES if j in joint_names]

        if self._ee_body_idx < 0:
            raise ValueError(f"Could not find body '{_EE_BODY_NAME}' in Franka.")
        if robot.is_fixed_base:
            self._jacobi_body_idx = self._ee_body_idx - 1
            self._jacobi_joint_ids = self._arm_joint_ids
        else:
            self._jacobi_body_idx = self._ee_body_idx
            self._jacobi_joint_ids = [jid + 6 for jid in self._arm_joint_ids]

        self._rest_joint_pos = torch.zeros(env.num_envs, len(joint_names), device=env.device)
        for idx, name in enumerate(joint_names):
            if name in _FRANKA_REST_JOINT_POS:
                self._rest_joint_pos[:, idx] = _FRANKA_REST_JOINT_POS[name]

        robot.write_joint_state_to_sim(
            position=self._rest_joint_pos,
            velocity=torch.zeros_like(self._rest_joint_pos),
        )
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        self._rest_ee_pos_w = self._ee_pos_w(robot).clone()

    def check_success(self, env) -> bool:
        storage_pos = env.scene[_STORAGE_BOX_NAME].data.root_pos_w - env.scene.env_origins
        done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for obj_name in _OBJECT_NAMES:
            obj_pos = env.scene[obj_name].data.root_pos_w - env.scene.env_origins
            done = torch.logical_and(done, obj_pos[:, 0] < storage_pos[:, 0] + _SUCCESS_X_RANGE[1])
            done = torch.logical_and(done, obj_pos[:, 0] > storage_pos[:, 0] + _SUCCESS_X_RANGE[0])
            done = torch.logical_and(done, obj_pos[:, 1] < storage_pos[:, 1] + _SUCCESS_Y_RANGE[1])
            done = torch.logical_and(done, obj_pos[:, 1] > storage_pos[:, 1] + _SUCCESS_Y_RANGE[0])
            done = torch.logical_and(done, obj_pos[:, 2] < storage_pos[:, 2] + _SUCCESS_Z_RANGE[1])
            done = torch.logical_and(done, obj_pos[:, 2] > storage_pos[:, 2] + _SUCCESS_Z_RANGE[0])
        return bool(done.all().item())

    def pre_step(self, env) -> None:
        pass

    def get_action(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)

        device = env.device
        num_envs = env.num_envs

        obj_name = _OBJECT_NAMES[self._current_object_idx]
        obj_pos_w = env.scene[obj_name].data.root_pos_w.clone()
        obj_quat_w = env.scene[obj_name].data.root_quat_w.clone()
        box_pos_w = env.scene[_STORAGE_BOX_NAME].data.root_pos_w.clone()
        robot_root_pos_w = robot.data.root_pos_w.clone()

        if self._step_count == 0 and self._event == 0:
            self._initial_ee_pos_w = self._ee_pos_w(robot).clone()

        phase_in_cycle = self._event % _PHASES_PER_OBJECT

        target_quat_w = self._gripper_down_quat_w(
            obj_quat_w,
            num_envs,
            device,
            obj_quat_w.dtype,
            yaw_offset=_GRASP_YAW_OFFSET,
        )

        grasp_anchor_w = _retreat_xy_toward(
            obj_pos_w,
            robot_root_pos_w,
            _GRASP_RETREAT_PER_OBJECT.get(obj_name, 0.0),
        )
        grasp_dx, grasp_dy = _GRASP_XY_OFFSET_PER_OBJECT.get(obj_name, (0.0, 0.0))
        grasp_anchor_w[:, 0] += grasp_dx
        grasp_anchor_w[:, 1] += grasp_dy

        drop_dx, drop_dy = _DROP_XY_OFFSET_PER_OBJECT.get(obj_name, (0.0, 0.0))
        drop_pos_w = box_pos_w.clone()
        drop_pos_w[:, 0] += drop_dx
        drop_pos_w[:, 1] += drop_dy

        if phase_in_cycle == 0:
            target_pos_w, gripper_cmd = self._phase_move_above_object(obj_pos_w, num_envs, device)
        elif phase_in_cycle == 1:
            target_pos_w, gripper_cmd = self._phase_approach_object(grasp_anchor_w, num_envs, device)
        elif phase_in_cycle == 2:
            target_pos_w, gripper_cmd = self._phase_grasp(
                grasp_anchor_w,
                num_envs,
                device,
                z_offset=_GRASP_Z_AT_CLOSE_PER_OBJECT.get(obj_name, 0.0),
            )
        elif phase_in_cycle == 3:
            target_pos_w, gripper_cmd = self._phase_lift(obj_pos_w, num_envs, device)
        elif phase_in_cycle == 4:
            target_pos_w, gripper_cmd = self._phase_move_above_box(drop_pos_w, num_envs, device)
        elif phase_in_cycle == 5:
            target_pos_w, gripper_cmd = self._phase_lower_to_release(drop_pos_w, num_envs, device)
        else:
            target_pos_w, gripper_cmd = self._phase_retreat(drop_pos_w, num_envs, device)

        # Cache current target, EE, object, and finger positions for advance() checks.
        self._last_target_pos_w = target_pos_w.clone()
        self._last_ee_pos_w = self._ee_pos_w(robot).clone()
        self._last_obj_pos_w = obj_pos_w.clone()
        if self._finger_joint_ids:
            self._last_finger_pos = robot.data.joint_pos[:, self._finger_joint_ids].clone()

        return self._joint_position_franka_action(env, target_pos_w, target_quat_w, gripper_cmd)

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _phase_move_above_object(self, obj_pos_w, num_envs, device):
        target = obj_pos_w.clone()
        target[:, 2] += _HOVER_Z_OFFSET
        if self._initial_ee_pos_w is not None:
            denom = max(self._events_dt[self._event] - 1, 1)
            alpha = min(self._step_count / denom, 1.0)
            target = (1.0 - alpha) * self._initial_ee_pos_w + alpha * target
        return target, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    def _phase_approach_object(self, obj_pos_w, num_envs, device):
        target = obj_pos_w.clone()
        target[:, 2] += _GRASP_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    def _phase_grasp(self, obj_pos_w, num_envs, device, z_offset: float):
        target = obj_pos_w.clone()
        target[:, 2] += z_offset
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_lift(self, obj_pos_w, num_envs, device):
        target = obj_pos_w.clone()
        target[:, 2] += _LIFT_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_move_above_box(self, box_pos_w, num_envs, device):
        target = box_pos_w.clone()
        target[:, 2] += _LIFT_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_lower_to_release(self, box_pos_w, num_envs, device):
        target = box_pos_w.clone()
        target[:, 2] += _RELEASE_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_retreat(self, box_pos_w, num_envs, device):
        target = box_pos_w.clone()
        target[:, 2] += _LIFT_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def _do_advance_phase(self) -> None:
        """Move to the next phase, clearing per-phase and per-object caches."""
        self._event += 1
        self._step_count = 0
        self._phase_convergence_count = 0
        self._last_target_pos_w = None
        self._last_ee_pos_w = None
        self._last_obj_pos_w = None
        self._last_finger_pos = None

        if self._event >= len(self._events_dt):
            self._episode_done = True
            return

        new_obj_idx = self._event // _PHASES_PER_OBJECT
        if new_obj_idx != self._current_object_idx:
            self._current_object_idx = new_obj_idx
            self._initial_ee_pos_w = None
            self._gripper_down_yaw_w = None
            self._gripper_down_yaw_offset_w = None

    def advance(self) -> None:
        if self._episode_done:
            return

        self._step_count += 1
        phase_in_cycle = self._event % _PHASES_PER_OBJECT

        # Per-phase early-exit (only after minimum steps, only for non-fixed phases).
        if (
            phase_in_cycle not in _FIXED_DURATION_PHASES
            and self._step_count >= _PHASE_MIN_STEPS[phase_in_cycle]
        ):
            if phase_in_cycle == 1:
                # Approach → Grasp: measure EE distance to the actual block
                # centre, not the computed approach target.  This is more
                # reliable because retreat/xy offsets in the target can
                # accumulate error.  Use per-object hold steps so that objects
                # requiring more precise positioning (e.g. the blue cylinder)
                # wait longer for the IK to settle before transitioning.
                obj_name = _OBJECT_NAMES[self._current_object_idx]
                hold_required = _APPROACH_HOLD_STEPS_PER_OBJECT.get(
                    obj_name, _CONVERGENCE_HOLD_STEPS
                )
                xy_threshold = _APPROACH_XY_THRESHOLD_PER_OBJECT.get(obj_name, 0.05)
                if self._last_ee_pos_w is not None and self._last_obj_pos_w is not None:
                    diff = self._last_ee_pos_w - self._last_obj_pos_w
                    dist_3d = torch.linalg.norm(diff, dim=-1).max().item()
                    dist_xy = torch.linalg.norm(diff[:, :2], dim=-1).max().item()
                    if dist_3d < _EE_TO_BLOCK_THRESHOLD and dist_xy < xy_threshold:
                        self._phase_convergence_count += 1
                        if self._phase_convergence_count >= hold_required:
                            self._do_advance_phase()
                            return
                    else:
                        self._phase_convergence_count = 0

            elif phase_in_cycle == 2:
                # Grasp → Lift: fingers must stay open (object held) for
                # _CONVERGENCE_HOLD_STEPS consecutive steps.
                if self._last_finger_pos is not None:
                    total_width = self._last_finger_pos.sum(dim=-1).max().item()
                    if total_width > _MIN_GRASP_WIDTH:
                        self._phase_convergence_count += 1
                        if self._phase_convergence_count >= _CONVERGENCE_HOLD_STEPS:
                            self._do_advance_phase()
                            return
                    else:
                        self._phase_convergence_count = 0

            elif phase_in_cycle == 3:
                # Lift → Move-above-box: object has risen to target height.
                if self._last_obj_pos_w is not None:
                    obj_z = self._last_obj_pos_w[0, 2].item()
                    if obj_z > _LIFT_SUCCESS_Z:
                        self._do_advance_phase()
                        return

            elif phase_in_cycle == 4:
                # Move-above-box → Lower: EE must be stably above the drop
                # position before the robot starts descending.  Uses the same
                # consecutive-step check as approach to prevent false positives.
                if self._last_target_pos_w is not None and self._last_ee_pos_w is not None:
                    dist = torch.linalg.norm(
                        self._last_ee_pos_w - self._last_target_pos_w, dim=-1
                    ).max().item()
                    if dist < _EE_CONVERGENCE_THRESHOLD:
                        self._phase_convergence_count += 1
                        if self._phase_convergence_count >= _CONVERGENCE_HOLD_STEPS:
                            self._do_advance_phase()
                            return
                    else:
                        self._phase_convergence_count = 0

            elif phase_in_cycle == 6:
                # Retreat → next object: EE converged to retreat target.
                if self._last_target_pos_w is not None and self._last_ee_pos_w is not None:
                    dist = torch.linalg.norm(
                        self._last_ee_pos_w - self._last_target_pos_w, dim=-1
                    ).max().item()
                    if dist < _EE_CONVERGENCE_THRESHOLD:
                        self._do_advance_phase()
                        return

        # Grasp failure detection: during lift phase, abort if object hasn't risen.
        if (
            phase_in_cycle == 3
            and self._step_count == _GRASP_CHECK_STEP
            and self._last_obj_pos_w is not None
        ):
            obj_z = self._last_obj_pos_w[0, 2].item()
            if obj_z < _MIN_LIFT_Z:
                self._episode_done = True
                return

        # Fall back to fixed-duration advancement.
        if self._step_count >= self._events_dt[self._event]:
            self._do_advance_phase()

    def reset(self) -> None:
        self._step_count = 0
        self._episode_done = False
        self._event = 0
        self._current_object_idx = 0
        self._initial_ee_pos_w = None
        self._gripper_down_yaw_w = None
        self._gripper_down_yaw_offset_w = None
        self._phase_convergence_count = 0
        self._last_target_pos_w = None
        self._last_ee_pos_w = None
        self._last_obj_pos_w = None
        self._last_finger_pos = None

    # ------------------------------------------------------------------
    # IK / control helpers (same as CupStackingStateMachine)
    # ------------------------------------------------------------------

    def _ee_pos_w(self, robot) -> torch.Tensor:
        body_idx = self._ee_body_idx if self._ee_body_idx >= 0 else -1
        return robot.data.body_pos_w[:, body_idx, :]

    def _ee_quat_w(self, robot) -> torch.Tensor:
        body_idx = self._ee_body_idx if self._ee_body_idx >= 0 else -1
        return robot.data.body_quat_w[:, body_idx, :]

    def _joint_position_franka_action(
        self,
        env,
        target_pos_w: torch.Tensor,
        target_quat_w: torch.Tensor,
        gripper_cmd: torch.Tensor,
    ) -> torch.Tensor:
        robot = env.scene["robot"]
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        root_quat_inv = quat_inv(root_quat_w)

        target_pos_root = quat_apply(root_quat_inv, target_pos_w - root_pos_w)
        ee_pos_root = quat_apply(root_quat_inv, self._ee_pos_w(robot) - root_pos_w)
        delta_pos_root = _clamp_delta(target_pos_root - ee_pos_root)

        delta_quat_w = _shortest_quat(quat_mul(target_quat_w, quat_inv(self._ee_quat_w(robot))))
        delta_rot_w = axis_angle_from_quat(delta_quat_w)
        delta_rot_root = _clamp_delta(quat_apply(root_quat_inv, delta_rot_w), _MAX_ROT_DELTA)

        pose_delta_root = torch.cat([delta_pos_root, delta_rot_root], dim=-1)
        joint_pos_target = self._arm_joint_pos(robot) + self._compute_delta_joint_pos(
            pose_delta_root, self._ee_jacobian_root(robot)
        )
        joint_pos_target = self._clamp_arm_joint_pos(robot, joint_pos_target)
        return torch.cat([joint_pos_target, gripper_cmd], dim=-1)

    def _arm_joint_pos(self, robot) -> torch.Tensor:
        if not self._arm_joint_ids:
            raise RuntimeError("setup() must run before requesting actions.")
        return robot.data.joint_pos[:, self._arm_joint_ids]

    def _ee_jacobian_root(self, robot) -> torch.Tensor:
        if self._jacobi_body_idx < 0 or not self._jacobi_joint_ids:
            raise RuntimeError("setup() must run before requesting actions.")

        jacobian = robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ].clone()
        root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
        jacobian[:, :3, :] = torch.bmm(root_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _compute_delta_joint_pos(self, pose_delta: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        jacobian_t = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (_IK_DLS_LAMBDA**2) * torch.eye(
            jacobian.shape[1], device=jacobian.device, dtype=jacobian.dtype
        )
        delta_joint_pos = (
            jacobian_t @ torch.inverse(jacobian @ jacobian_t + lambda_matrix) @ pose_delta.unsqueeze(-1)
        )
        return delta_joint_pos.squeeze(-1)

    def _clamp_arm_joint_pos(self, robot, joint_pos: torch.Tensor) -> torch.Tensor:
        joint_pos_limits = getattr(robot.data, "soft_joint_pos_limits", None)
        if joint_pos_limits is None:
            joint_pos_limits = getattr(robot.data, "joint_pos_limits", None)
        if joint_pos_limits is None:
            return joint_pos
        arm_joint_pos_limits = joint_pos_limits[:, self._arm_joint_ids, :]
        return torch.clamp(joint_pos, arm_joint_pos_limits[..., 0], arm_joint_pos_limits[..., 1])

    def _gripper_down_quat_w(
        self,
        obj_quat_w: torch.Tensor,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
        yaw_offset: float = 0.0,
    ) -> torch.Tensor:
        if self._gripper_down_yaw_w is None or self._gripper_down_yaw_w.shape[0] != num_envs:
            base_yaw = _yaw_from_quat_wxyz(obj_quat_w).to(device=device, dtype=dtype)
            self._gripper_down_yaw_offset_w = torch.empty(num_envs, device=device, dtype=dtype).uniform_(
                _GRIPPER_DOWN_YAW_OFFSET_RANGE[0],
                _GRIPPER_DOWN_YAW_OFFSET_RANGE[1],
            )
            self._gripper_down_yaw_w = (
                base_yaw + yaw_offset + self._gripper_down_yaw_offset_w
            ).clone()

        roll = torch.full((num_envs,), _GRIPPER_DOWN_ROLL_W, device=device, dtype=dtype)
        pitch = torch.full((num_envs,), _GRIPPER_DOWN_PITCH_W, device=device, dtype=dtype)
        yaw = self._gripper_down_yaw_w.to(device=device, dtype=dtype)
        return quat_from_euler_xyz(roll, pitch, yaw)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def task_object_names(self) -> tuple[str, ...]:
        return _OBJECT_NAMES + (_STORAGE_BOX_NAME,)

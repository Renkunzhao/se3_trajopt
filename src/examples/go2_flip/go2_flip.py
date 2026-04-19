#!/usr/bin/env python3
"""Shared utilities for Go2 flip examples aligned with quad_sideflip.py."""

import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("IPOPT_OUTPUT_DIR", os.path.join(_REPO_ROOT, "ipopt_output"))

from nltrajopt.trajectory_optimization import NLTrajOpt
from nltrajopt.contact_scheduler import ContactScheduler
from nltrajopt.node import Node
from nltrajopt.constraint_models import (
    WholeBodyDynamics,
    TimeConstraint,
    SemiEulerIntegration,
    TerrainGridContactConstraints,
    TerrainGridFrictionConstraints,
)
from nltrajopt.cost_models import ConfigurationCost, JointVelocityCost, JointAccelerationCost
import nltrajopt.params as pars
import nltrajopt.utils as reprutils
from robots.go2.Go2Wrapper import Go2
from terrain.terrain_grid import TerrainGrid
from visualiser.visualiser import TrajoptVisualiser

DT = 0.02
TARGET_DELTA = 0.3

CSV_JOINT_ORDER = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]


def q_pin_to_row(model: pin.Model, q: np.ndarray) -> np.ndarray:
    pos = q[0:3]
    quat = q[3:7]
    joints = np.array(
        [q[model.joints[model.getJointId(jname)].idx_q] for jname in CSV_JOINT_ORDER]
    )
    return np.hstack([pos, quat, joints])


def v_pin_to_row(model: pin.Model, v: np.ndarray) -> np.ndarray:
    base = v[0:6]
    joints = np.array(
        [v[model.joints[model.getJointId(jname)].idx_v] for jname in CSV_JOINT_ORDER]
    )
    return np.hstack([base, joints])


def tau_to_row(model: pin.Model, tau: np.ndarray) -> np.ndarray:
    return np.array(
        [tau[model.joints[model.getJointId(jname)].idx_v] for jname in CSV_JOINT_ORDER]
    )


def _compute_tau(model: pin.Model, data: pin.Data, node: dict) -> np.ndarray:
    q, v, a = node["q"], node["v"], node["a"]
    fext = [pin.Force.Zero() for _ in range(model.njoints)]
    for frame, force in node["forces"].items():
        frame_id = model.getFrameId(frame)
        frame_data = model.frames[frame_id]
        jMf = frame_data.placement
        f_world = pin.Force(force, np.zeros(3))
        fext[frame_data.parentJoint] += jMf.act(f_world)
    pin.forwardKinematics(model, data, q, v, a)
    pin.updateFramePlacements(model, data)
    return pin.rnea(model, data, q, v, a, fext)


def export_csv(model: pin.Model, nodes: list, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    q_cols = ["x", "y", "z", "qx", "qy", "qz", "qw"] + CSV_JOINT_ORDER
    v_cols = ["vx", "vy", "vz", "wx", "wy", "wz"] + [f"d{j}" for j in CSV_JOINT_ORDER]
    tau_cols = [f"tau_{j}" for j in CSV_JOINT_ORDER]
    header = ",".join(q_cols + v_cols + tau_cols)

    data = model.createData()
    rows = []
    for node in nodes:
        tau = _compute_tau(model, data, node)
        rows.append(
            np.hstack(
                [
                    q_pin_to_row(model, node["q"]),
                    v_pin_to_row(model, node["v"]),
                    tau_to_row(model, tau),
                ]
            )
        )
    rows = np.array(rows)
    np.savetxt(output_path, rows, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"[TO] 已保存 {len(rows)} 帧 → {output_path}")


def load_config(xml_path: str) -> dict:
    """Parse a go2_flip XML config file and return a dict of typed values."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def _text(tag, default=None):
        el = root.find(tag)
        if el is None:
            if default is not None:
                return default
            raise KeyError(f"Missing required tag <{tag}> in {xml_path}")
        return el.text.strip()

    return {
        "flip_label":     _text("flip_label"),
        "angle_axis":     int(_text("angle_axis")),
        "angle_total":    float(_text("angle_total")),
        "target_pos_axis": int(_text("target_pos_axis")),
        "target_delta":   float(_text("target_delta")),
        "pre_flip_t":     float(_text("pre_flip_t", "1.0")),
        "flight_t":       float(_text("flight_t", "0.4")),
        "land_t":         float(_text("land_t", "1.0")),
        "dt":             float(_text("dt", str(DT))),
        "max_iter":       int(_text("max_iter", "200")),
        "tol":            float(_text("tol", "1e-3")),
        "output":         _text("output", "export/go2_flip.csv"),
        "solution_name":  _text("solution_name"),
        "vis":            _text("vis", "false").lower() in ("true", "1", "yes"),
        # costs (0 = disabled)
        "w_config":        float(_text("w_config", "0")),
        "w_velocity":      float(_text("w_velocity", "0")),
        "w_acceleration":  float(_text("w_acceleration", "0")),
        "w_flight_config":   float(_text("w_flight_config", "0")),
        "w_flight_velocity": float(_text("w_flight_velocity", "0")),
    }


def run_go2_flip(xml_path: str) -> None:
    cfg = load_config(xml_path)

    flip_label     = cfg["flip_label"]
    angle_axis     = cfg["angle_axis"]
    angle_total    = cfg["angle_total"]
    target_pos_axis = cfg["target_pos_axis"]
    target_delta   = cfg["target_delta"]
    dt             = cfg["dt"]

    robot = Go2()
    q = robot.go_neutral()

    terrain = TerrainGrid(10, 10, 0.9, -1.0, -5.0, 5.0, 5.0)
    terrain.set_zero()

    contacts_dict = {
        "l_foot": robot.left_foot_frames,
        "r_foot": robot.right_foot_frames,
        "l_gripper": robot.left_gripper_frames,
        "r_gripper": robot.right_gripper_frames,
    }

    contact_scheduler = ContactScheduler(
        robot.model, dt=dt, contact_frame_dict=contacts_dict
    )
    contact_scheduler.add_phase(
        ["l_foot", "r_foot", "l_gripper", "r_gripper"], cfg["pre_flip_t"]
    )
    k1 = len(contact_scheduler.contact_sequence_fnames)
    contact_scheduler.add_phase([], cfg["flight_t"])
    k2 = len(contact_scheduler.contact_sequence_fnames)
    contact_scheduler.add_phase(
        ["l_foot", "r_foot", "l_gripper", "r_gripper"], cfg["land_t"]
    )

    frame_contact_seq = contact_scheduler.contact_sequence_fnames
    contact_frame_names = (
        robot.left_foot_frames
        + robot.right_foot_frames
        + robot.left_gripper_frames
        + robot.right_gripper_frames
    )
    # Additional frames for ground-penetration avoidance (p_z >= 0)
    collision_frames = [
        "base",
        "FL_hip", "FR_hip", "RL_hip", "RR_hip",
        "FL_calf", "FR_calf", "RL_calf", "RR_calf",
        "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    ]
    contact_frame_names = contact_frame_names + collision_frames

    K = len(frame_contact_seq)
    print("K = ", K)

    stages = []
    for k, contact_phase_fnames in enumerate(frame_contact_seq):
        stage_node = Node(
            nv=robot.model.nv,
            contact_phase_fnames=contact_phase_fnames,
            contact_fnames=contact_frame_names,
        )

        dyn_const = WholeBodyDynamics()
        stage_node.dynamics_type = dyn_const.name
        stage_node.constraints_list.extend(
            [
                dyn_const,
                TimeConstraint(min_dt=dt, max_dt=dt, total_time=None),
                SemiEulerIntegration(),
                TerrainGridContactConstraints(terrain),
                TerrainGridFrictionConstraints(terrain),
            ]
        )

        nq_joint = robot.model.nq - 7
        nv_joint = robot.model.nv - 6
        if cfg["w_config"] > 0:
            stage_node.costs_list.append(
                ConfigurationCost(q.copy()[7:], np.eye(nq_joint) * cfg["w_config"])
            )
        if cfg["w_velocity"] > 0:
            stage_node.costs_list.append(
                JointVelocityCost(np.zeros(nv_joint), np.eye(nv_joint) * cfg["w_velocity"])
            )
        if cfg["w_acceleration"] > 0:
            stage_node.costs_list.append(
                JointAccelerationCost(np.zeros(nv_joint), np.eye(nv_joint) * cfg["w_acceleration"])
            )
        if cfg["w_flight_config"] > 0 and k1 <= k < k2:
            stage_node.costs_list.append(
                ConfigurationCost(q.copy()[7:], np.eye(nq_joint) * cfg["w_flight_config"])
            )
        if cfg["w_flight_velocity"] > 0 and k1 <= k < k2:
            stage_node.costs_list.append(
                JointVelocityCost(np.zeros(nv_joint), np.eye(nv_joint) * cfg["w_flight_velocity"])
            )

        stages.append(stage_node)

    opti = NLTrajOpt(model=robot.model, nodes=stages, dt=dt)
    opti.set_initial_pose(q)
    qf = np.copy(q)
    qf[target_pos_axis] = target_delta
    opti.set_target_pose(qf)

    for k, node in enumerate(opti.nodes):
        if k1 <= k <= k2:
            theta = angle_total * (k - k1) / max(k2 - k1, 1)
            rpy = [0.0, 0.0, 0.0]
            rpy[angle_axis] = theta
            opti.x0[node.q_id] = reprutils.rpy2rep(q, rpy)

    print(f"[TO] 开始求解 ({flip_label}, IPOPT)...")
    result = opti.solve(cfg["max_iter"], cfg["tol"], True, print_level=0)
    opti.save_solution(cfg["solution_name"])
    export_csv(robot.model, result["nodes"], cfg["output"])

    K = len(result["nodes"])
    dts = [result["nodes"][k]["dt"] for k in range(K)]
    qs = [result["nodes"][k]["q"] for k in range(K)]
    forces = [result["nodes"][k]["forces"] for k in range(K)]

    if cfg["vis"] or pars.VIS:
        tvis = TrajoptVisualiser(robot)
        tvis.display_robot_q(robot, qs[0])

        time.sleep(1.0)
        while True:
            for i in range(len(qs)):
                time.sleep(dts[i])
                tvis.display_robot_q(robot, qs[i])
                tvis.update_forces(robot, forces[i], 0.01)
            tvis.update_forces(robot, {}, 0.01)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.xml>")
        sys.exit(1)
    run_go2_flip(sys.argv[1])

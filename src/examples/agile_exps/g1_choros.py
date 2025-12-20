import time
import argparse
import numpy as np
import pinocchio as pin
import nltrajopt.params as pars
import nltrajopt.utils as reprutils
from nltrajopt.trajectory_optimization import NLTrajOpt
from nltrajopt.contact_scheduler import ContactScheduler
from nltrajopt.node import Node
from nltrajopt.constraint_models import *
from nltrajopt.cost_models import *
from terrain.terrain_grid import TerrainGrid
from robots.g1.G1Wrapper import G1
from visualiser.visualiser import TrajoptVisualiser


parser = argparse.ArgumentParser(description = "Solve optimization and Visualize the result")
parser.add_argument(
    "--vis",
    action = "store_true",
    help = "Enable Visualization",
)
args = parser.parse_args()

np.set_printoptions(precision=2, suppress=False)

robot = G1()




########## modify these ##########

motion_type = "g1_walk"
motion_number = 0
DT = 0.05
xf = 0.5
yf = 0.1

rows = 10
cols = 10
mu = 0.3
min_x = -1.0
max_x = 5.0
min_y = -5.0
max_y = 5.0

# phases = [
#     [["l_foot", "r_foot"], 1.0],
#     [[], 0.5],
#     [["l_foot", "r_foot"], 1.0],
# ]

phases = [
    [["l_foot", "r_foot"], 1.0],
]
for _ in range(3):
    phases += [
        [["l_foot"], 0.4],
        [["l_foot", "r_foot"], 0.3],
        [["r_foot"], 0.4],
        [["l_foot", "r_foot"], 0.3],
    ]
phases += [
    [["l_foot", "r_foot"], 1.0],
]

########## modify these ##########




terrain = TerrainGrid(rows, cols, mu, min_x, min_y, max_x, max_y)
terrain.set_zero()

contacts_dict = {
    "l_foot": robot.left_foot_frames,
    "r_foot": robot.right_foot_frames,
    "l_gripper": robot.left_gripper_frames,
    "r_gripper": robot.right_gripper_frames,
}
contact_scheduler = ContactScheduler(robot.model, dt = DT, contact_frame_dict = contacts_dict)

q = robot.go_neutral()
qf = np.copy(q)
qf[0] = xf
qf[1] = yf

for phase in phases:
    contact_scheduler.add_phase(phase[0], phase[1])

frame_contact_seq = contact_scheduler.contact_sequence_fnames
contact_frame_names = robot.left_foot_frames + robot.right_foot_frames + robot.left_gripper_frames + robot.right_gripper_frames

stages = []
K = len(frame_contact_seq)
print("K = ", K)
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
            TimeConstraint(min_dt=DT, max_dt=DT, total_time=None),
            SemiEulerIntegration(),
            TerrainGridContactConstraints(terrain),
            TerrainGridFrictionConstraints(terrain, max_delta_force=20),
        ]
    )

    stage_node.costs_list.extend([ConfigurationCost(q.copy()[7:], np.eye(robot.model.nq - 7) * 1e-6)])
    # stage_node.costs_list.extend([JointVelocityCost(np.zeros((robot.model.nv - 6,)), np.eye(robot.model.nv - 6) * 1e-3)])
    stage_node.costs_list.extend([JointAccelerationCost(np.zeros((robot.model.nv - 6,)), np.eye(robot.model.nv - 6) * 1e-6)])

    stages.append(stage_node)

opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)

opti.set_initial_pose(q)
opti.set_target_pose(qf)

# warm start
# for k, node in enumerate(opti.nodes):
#     if k1 <= k <= k2:
#         theta = -2 * np.pi * (k - k1) / (k2 - k1)
#         opti.x0[node.q_id] = reprutils.rpy2rep(q, [0.0, theta, 0.0])

result = opti.solve(200, 1e-3, False, print_level=0)
opti.save_solution(motion_type, motion_number)

K = len(result["nodes"])
dts = [result["nodes"][k]["dt"] for k in range(K)]
qs = [result["nodes"][k]["q"] for k in range(K)]
forces = [result["nodes"][k]["forces"] for k in range(K)]

if pars.VIS:
    tvis = TrajoptVisualiser(robot)
    tvis.display_robot_q(robot, qs[0])
    time.sleep(1)
    while True:
        for i in range(len(qs)):
            time.sleep(dts[i])
            tvis.display_robot_q(robot, qs[i])
            tvis.update_forces(robot, forces[i], 0.01)
        tvis.update_forces(robot, {}, 0.01)

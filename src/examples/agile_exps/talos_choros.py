import time
import argparse
import numpy as np
import pinocchio as pin
import nltrajopt.params as pars
from nltrajopt.trajectory_optimization import NLTrajOpt
from nltrajopt.contact_scheduler import ContactScheduler
from nltrajopt.node import Node
from nltrajopt.constraint_models import *
from nltrajopt.cost_models import *
from terrain.terrain_grid import TerrainGrid
from robots.talos.TalosWrapper import Talos
from visualiser.visualiser import TrajoptVisualiser


parser = argparse.ArgumentParser(description = "Solve optimization and Visualize the result")
parser.add_argument(
    "--vis",
    action = "store_true",
    help = "Enable Visualization",
)
args = parser.parse_args()

np.set_printoptions(precision=2, suppress=False)

robot = Talos()




########## modify these ##########

motion_type = "talos_walk"
motion_number = 0
DT = 0.05
xf = 3.0
yf = 1.0

rows = 10
cols = 10
mu = 0.9
min_x = -1.0
max_x = 5.0
min_y = -5.0
max_y = 5.0

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




# terrain
terrain = TerrainGrid(rows, cols, mu, min_x, min_y, max_x, max_y)
terrain.set_zero()


# contacts
contacts_dict = {
    "l_foot": robot.left_foot_frames,
    "r_foot": robot.right_foot_frames,
    "l_gripper": robot.left_gripper_frames,
    "r_gripper": robot.right_gripper_frames,
}

contact_scheduler = ContactScheduler(
    robot.model,
    dt=DT,
    contact_frame_dict=contacts_dict,
)


# initial / target configuration
q = robot.go_neutral()
qf = np.copy(q)
qf[0] = xf
qf[1] = yf
qf[2] += terrain.height(qf[0], qf[1])


# build contact sequence
for phase in phases:
    contact_scheduler.add_phase(phase[0], phase[1])

frame_contact_seq = contact_scheduler.contact_sequence_fnames

contact_frame_names = (
    robot.left_foot_frames
    + robot.right_foot_frames
    + robot.left_gripper_frames
    + robot.right_gripper_frames
)


# build trajectory nodes
stages = []
K = len(frame_contact_seq)
print("K =", K)

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
            TerrainGridFrictionConstraints(terrain),
        ]
    )

    stage_node.costs_list.extend(
        [
            ConfigurationCost(
                q.copy()[7:],
                np.eye(robot.model.nv - 6) * 1e-6,
            )
        ]
    )

    stages.append(stage_node)


# trajectory optimization
opti = NLTrajOpt(
    model=robot.model,
    nodes=stages,
    dt=DT,
)

opti.set_initial_pose(q)
opti.set_target_pose(qf)

result = opti.solve(50, 1e-3, parallel=False, print_level=5)
opti.save_solution(motion_type, motion_number)


# extract solution
K = len(result["nodes"])
dts = [result["nodes"][k]["dt"] for k in range(K)]
qs = [result["nodes"][k]["q"] for k in range(K)]
forces = [result["nodes"][k]["forces"] for k in range(K)]


# visualization
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

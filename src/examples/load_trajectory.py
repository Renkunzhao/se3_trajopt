import time
import os
import argparse
import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt
import nltrajopt.params as pars
from robots.talos.TalosWrapper import Talos
from robots.g1.G1Wrapper import G1
from robots.go2.Go2Wrapper import Go2
from nltrajopt.trajectory_optimization import NLTrajOpt
from nltrajopt.se3tangent import *
from visualiser.visualiser import TrajoptVisualiser


# --------------------------------------------------
# argument parsing
# --------------------------------------------------

parser = argparse.ArgumentParser(description = "Visualize a trajectory optimization result")

parser.add_argument(
    "--robot",
    type = str,
    choices = ["talos", "g1", "go2"],
    default = "g1",
    required = True,
    help = "Robot type",
)

parser.add_argument(
    "--motion_type",
    type = str,
    required = True,
    help = "Motion Type Name (folder name)",
)

parser.add_argument(
    "--motion_number",
    type = int,
    default = 0,
    required = True,
    help = "Motion Index Number",
)

parser.add_argument(
    "--vis",
    action = "store_true",
    help = "Enable Visualization",
)

args = parser.parse_args()


# --------------------------------------------------
# robot selection
# --------------------------------------------------

if args.robot == "talos":
    robot = Talos()
elif args.robot == "g1":
    robot = G1()
elif args.robot == "go2":
    robot = Go2()
else:
    raise ValueError(f"Unknown robot type: {args.robot}")

motion_type = args.motion_type
motion_number = args.motion_number

pars.VIS = args.vis


# --------------------------------------------------
# load trajectory
# --------------------------------------------------

TRAJ_PATH = f"trajopt_solutions_batch/{motion_type}/{motion_type}_{motion_number}.json"

if not os.path.exists(TRAJ_PATH):
    raise FileNotFoundError(f"Trajectory file not found: {TRAJ_PATH}")

results = NLTrajOpt.load_solution(TRAJ_PATH)

K = len(results["nodes"])
dts = [results["nodes"][k]["dt"] for k in range(K)]
qs = [results["nodes"][k]["q"] for k in range(K)]
forces = [results["nodes"][k]["forces"] for k in range(K)]


# --------------------------------------------------
# visualization
# --------------------------------------------------

print("Average frequency:", 1 / np.mean(dts))
print("Number of nodes:", K)

if pars.VIS:
    tvis = TrajoptVisualiser(robot)
    tvis.display_robot_q(robot, qs[0])
    time.sleep(1)
    try:
        while True:
            for i in range(len(qs)):
                time.sleep(dts[i])
                tvis.display_robot_q(robot, qs[i])
                tvis.update_forces(robot, forces[i], 0.01)
            tvis.update_forces(robot, {}, 0.01)
    except KeyboardInterrupt:
        print("End of visualization")

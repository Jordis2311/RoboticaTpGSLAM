import graphviz
import numpy as np
import plotly.graph_objects as go

import gtsam
from gtsam import Pose3
from gtsam.utils import plot
import matplotlib.pyplot as plt


def readData(file = 'parking-garage.g2o'):
    vertexes = []
    edges = []
    with open(f"data/{file}", 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if line[0] == 'VERTEX_SE3:QUAT':
                idx = int(line[1])
                x = float(line[2])
                y = float(line[3])
                z = float(line[4])
                quaternion = list(map(float, line[5:9]))
                vertexes.append((idx, x, y, z, quaternion))
            elif line[0] == 'EDGE_SE2':
                i = int(line[1])
                j = int(line[2])
                dx = float(line[3])
                dy = float(line[4])
                dz = float(line[5])
                quaternion = list(map(float, line[6:10]))
                information_values = list(map(float, line[10:]))
                edges.append((i, j, dx, dy, dz, quaternion, information_values))
    return vertexes, edges


# [[q[0], q[1], q[2]], q[3], q[4], q[5]]
#  [q[1], q[6], q[7]], q[8], q[9], q[10]]
#  [q[2], q[7], q[11]], q[12], q[13], q[14]]
#  [q[3], q[8], q[12]], q[15], q[16], q[17]]
#  [q[4], q[9], q[13]], q[16], q[18], q[19]]
#  [q[5], q[10], q[14]], q[17],q[19], q[20]]

def create_information_matrix_3d(q):
    information_matrix = np.array([[q[0], q[1], q[2], q[3], q[4], q[5]],
                                   [q[1], q[6], q[7], q[8], q[9], q[10]],
                                   [q[2], q[7], q[11], q[12], q[13], q[14]],
                                   [q[3], q[8], q[12], q[15], q[16], q[17]],
                                   [q[4], q[9], q[13], q[16], q[18], q[19]],
                                   [q[5], q[10], q[14], q[17], q[19], q[20]]])
    return information_matrix

def create_pose3(dx, dy, dz, quaternions):
    rotation3 = gtsam.Rot3.Quaternion(quaternions[0], quaternions[1], quaternions[2], quaternions[3])
    point3 = gtsam.Point3(dx, dy, dz)
    pose3 = gtsam.Pose3(rotation3, point3)
    return pose3

def createPoseGraph3D(vertexes, edges):
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()    
    for vertex in vertexes:
        idx, x, y, z, quaternions = vertex        
        pose3 = create_pose3(x, y, z, quaternions)
        if idx == 0:
            priorModel = gtsam.noiseModel.Diagonal.Variances(np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]))
            graph.add(gtsam.PriorFactorPose3(idx, pose3, priorModel))        
        initial_estimate.insert(idx, pose3)   

    for edge in edges:
        i, j, dx, dy, dz, quaternions, info = edge
        pose3 = create_pose3(dx, dy, dz, quaternions)
        information_matrix = create_information_matrix_3d(info)
        graph.add(gtsam.BetweenFactorPose3(i, j, pose3, information_matrix))

    return graph, initial_estimate

def optimizePoseGraph(graph, initial_estimate):
    parameters = gtsam.GaussNewtonParams()
    # Set optimization parameters
    parameters.setRelativeErrorTol(1e-5) # Stop when change in error is small
    parameters.setMaxIterations(100)     # Limit iterations
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, parameters)

    # Optimize!
    result = optimizer.optimize()
    return result, None
    # marginals = gtsam.Marginals(graph, result)
    # i = 1
    # covariances = []
    # while result.exists(i):
    #     covariances.append(marginals.marginalCovariance(i))
    #     i += 1
    # return result, covariances



def showComparisonGraphs3D(poses1, poses2, title1="Initial Trajectory", title2="Optimized Trajectory", output_path=None):
    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    plt.cla()

    # extract translation coordinates from both pose sets
    def extract_xyz(poses):
        xs, ys, zs = [], [], []
        for i in range(poses.size()):
            p = poses.atPose3(i)            
            xs.append(p.x()); ys.append(p.y()); zs.append(p.z())
        return xs, ys, zs

    xs1, ys1, zs1 = extract_xyz(poses1)
    xs2, ys2, zs2 = extract_xyz(poses2)

    # plot both trajectories
    ax.plot(xs1, ys1, zs1, color='tab:blue', linewidth=1.5, label=title1)
    ax.scatter(xs1, ys1, zs1, color='tab:blue', s=8)
    ax.plot(xs2, ys2, zs2, color='tab:orange', linewidth=1.5, label=title2)
    ax.scatter(xs2, ys2, zs2, color='tab:orange', s=8)

    # mark start and end points
    if xs1:
        ax.scatter(xs1[0], ys1[0], zs1[0], color='green', s=50, marker='o', label='start (initial)')
    if xs2:
        ax.scatter(xs2[-1], ys2[-1], zs2[-1], color='red', s=50, marker='X', label='end (optimized)')

    # attempt to set equal aspect for 3D plot (fallback provided)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        # fallback to manual equalization
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max(x_range, y_range, z_range) / 2.0
        x_mid = np.mean(x_limits)
        y_mid = np.mean(y_limits)
        z_mid = np.mean(z_limits)
        ax.set_xlim3d(x_mid - max_range, x_mid + max_range)
        ax.set_ylim3d(y_mid - max_range, y_mid + max_range)
        ax.set_zlim3d(z_mid - max_range, z_mid + max_range)

    ax.set_title(f"{title1} vs {title2}")
    ax.legend()
    ax.grid(False)

    if output_path:
        plt.savefig(f"pose3dImages/{output_path}.svg")
    else:
        plt.show()
    plt.close()
    plt.savefig(f"pose3dImages/{output_path}.svg")
    plt.close()

def showGraph3D(poses, cov=None, title="Initial Trajectory", output_path=None):    
    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    plt.cla()
    # Plot initial estimate poses
    for i in range(poses.size()):
        pose = poses.atPose3(i)
        cov_ = cov[i] if cov is not None else None      
        plot.plot_pose3(0, pose, axis_length=0.4, P = cov_)

    plt.axis('equal')    
    plt.title(title)
    # plt.xlabel("X-coordinate")
    # plt.ylabel("Y-coordinate")
    plt.grid(False)
    plt.savefig(f"pose3dImages/{output_path}.svg")
    plt.close() 


def main():
    
    print("Leyendo datos...\n")
    vertexes, edges = readData('parking-garage.g2o')
    
    print("Generando la estimacion inicial...\n")
    graph, initial_estimate = createPoseGraph3D(vertexes, edges)
    # print("Initial Estimate:", format(initial_estimate))
    # showGraph3D(initial_estimate, title="Initial 3D Pose Graph", output_path="initial_3d_pose_graph")

    print("Optimizando el grafo de poses con Gauss Newton...\n")
    optimizedGN, covariances = optimizePoseGraph(graph, initial_estimate)
    # showGraph3D(optimizedGN, cov=covariances, title="Optimized 3D Pose Graph", output_path="optimized_3d_pose_graph")
    showComparisonGraphs3D(initial_estimate, optimizedGN, title1="Unoptimized Trajectory", title2="Optimized Trajectory", output_path="3d_trajectory_comparison")



if __name__ == "__main__":
    main()
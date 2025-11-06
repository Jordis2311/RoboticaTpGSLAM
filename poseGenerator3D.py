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
            elif line[0] == 'EDGE_SE3:QUAT':
                i = int(line[1])
                j = int(line[2])
                dx = float(line[3])
                dy = float(line[4])
                dz = float(line[5])
                quaternion = list(map(float, line[6:10]))
                information_values = list(map(float, line[10:]))
                edges.append((i, j, dx, dy, dz, quaternion, information_values))
    return vertexes, edges


# [[q[0], q[1], q[2], q[3], q[4], q[5]]
#  [q[1], q[6], q[7], q[8], q[9], q[10]]
#  [q[2], q[7], q[11], q[12], q[13], q[14]]
#  [q[3], q[8], q[12], q[15], q[16], q[17]]
#  [q[4], q[9], q[13], q[16], q[18], q[19]]
#  [q[5], q[10], q[14], q[17], q[19], q[20]]

#       // g2o's EDGE_SE3:QUAT stores information/precision of Pose3 in t,R order, unlike GTSAM:
#   
# EDGE_SE3:QUAT 1332 1333 4.2216 -0.482494 0.00531712 -0.00313351 0.00563749 -0.115571 0.993278 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 4.00036 -0.000133256 0.0398948 3.99991 -0.00220671 3.94697

#       Matrix6 mgtsam;
#       mgtsam.block<3, 3>(0, 0) = m.block<3, 3>(3, 3); // info rotation
#       mgtsam.block<3, 3>(3, 3) = m.block<3, 3>(0, 0); // info translation
#       mgtsam.block<3, 3>(3, 0) = m.block<3, 3>(0, 3); // off diagonal g2o t,R -> GTSAM R,t
#       mgtsam.block<3, 3>(0, 3) = m.block<3, 3>(3, 0); // off diagonal g2o R,t -> GTSAM t,R
#       SharedNoiseModel model = noiseModel::Gaussian::Information(mgtsam);


def create_information_matrix_3d(q):
    # M = np.zeros((6, 6))
    # idx = np.triu_indices(6)
    # M[idx] = q
    # M = M + np.triu(M, 1).T
    # # print("Matriz original (g2o):\n", M)

    # # 2. Reordenar bloques (g2o -> GTSAM)
    # mgtsam = np.zeros((6, 6))
    # mgtsam[0:3, 0:3] = M[3:6, 3:6]
    # mgtsam[3:6, 3:6] = M[0:3, 0:3]
    # mgtsam[3:6, 0:3] = M[0:3, 3:6]
    # mgtsam[0:3, 3:6] = M[3:6, 0:3]
    mgtsam = np.array([[q[0], q[1], q[2], q[3], q[4], q[5]],
                                   [q[1], q[6], q[7], q[8], q[9], q[10]],
                                   [q[2], q[7], q[11], q[12], q[13], q[14]],
                                   [q[3], q[8], q[12], q[15], q[16], q[17]],
                                   [q[4], q[9], q[13], q[16], q[18], q[19]],
                                   [q[5], q[10], q[14], q[17], q[19], q[20]]])

    return gtsam.noiseModel.Gaussian.Information(mgtsam)

def create_pose3(dx, dy, dz, quaternions):
    rotation3 = gtsam.Rot3.Quaternion(quaternions[3], quaternions[0], quaternions[1], quaternions[2])
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
    parameters.setVerbosity("Termination")    
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, parameters)

    # Optimize!
    result = optimizer.optimize()    
    
    marginals = gtsam.Marginals(graph, result)
    covariances = [marginals.marginalCovariance(i) for i in range(result.size())]
    return result, covariances



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
    
    ax.plot(xs1, ys1, zs1, c='blue', label=title1, alpha=0.7, linewidth=0.75)
    sc2 = ax.plot(xs2, ys2, zs2, c='red', label=title2, alpha=0.7, linewidth=0.75)        
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')
    ax.set_title(f'{title1} vs {title2}')
    ax.legend()

    plt.savefig(f'pose3dImages/{output_path}.svg')
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
    # showGraph3D(initial_estimate, title="Initial 3D Pose Graph", output_path="initial_3d_pose_graph")
    # marginals = gtsam.Marginals(graph, initial_estimate)
    # showGraph3D(initial_estimate, cov=[marginals.marginalCovariance(i) for i in range(initial_estimate.size())],  output_path='pose_graph_initial_withCov')

    print("Optimizando el grafo de poses con Gauss Newton...\n")
    optimizedGN, covariances = optimizePoseGraph(graph, initial_estimate)
    showGraph3D(optimizedGN, title="Optimized 3D Pose Graph", output_path="optimized_3d_pose_graph")
    showComparisonGraphs3D(initial_estimate, optimizedGN, title1="Unoptimized Trajectory", title2="Optimized Trajectory", output_path="3d_trajectory_comparison")



if __name__ == "__main__":
    main()  
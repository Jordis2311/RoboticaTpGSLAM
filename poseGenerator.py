import graphviz
import numpy as np
import plotly.graph_objects as go

import gtsam
from gtsam import Pose2
from gtsam.utils import plot
import matplotlib.pyplot as plt


# [VERTEX_SE2 i x y theta] 
# [EDGE_SE2 i j dx dy dtheta q11 q12 q13 q22 q23 q33]
def readData(file = 'input_INTEL_g2o.g2o'):
    vertexes = []
    edges = []
    with open(f"data/{file}", 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if line[0] == 'VERTEX_SE2':
                idx = int(line[1])
                x = float(line[2])
                y = float(line[3])
                theta = float(line[4])
                vertexes.append(('VERTEX_SE2', idx, x, y, theta))
            elif line[0] == 'EDGE_SE2':
                i = int(line[1])
                j = int(line[2])
                dx = float(line[3])
                dy = float(line[4])
                dtheta = float(line[5])
                q = list(map(float, line[6:12]))
                edges.append(('EDGE_SE2', i, j, dx, dy, dtheta, q))
    return vertexes, edges


def createPoseGraph(vertexes, edges):
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()    
    for vertex in vertexes:
        _, idx, x, y, theta = vertex
        if idx == 0:
            graph.add(gtsam.PriorFactorPose2(idx, Pose2(x, y, theta), gtsam.noiseModel.Diagonal.Variances(np.array([1e-6, 1e-6, 1e-8]))))
        pose = Pose2(x, y, theta)
        initial_estimate.insert(idx, pose)   

    for edge in edges:
        _, i, j, dx, dy, dtheta, q = edge
        pose = Pose2(dx, dy, dtheta)
        information_matrix = gtsam.noiseModel.Gaussian.Information(np.array([[q[0], q[1], q[2]],
                                                                             [q[1], q[3], q[4]],
                                                                             [q[2], q[4], q[5]]]))
        graph.add(gtsam.BetweenFactorPose2(i, j, pose, information_matrix))

    return graph, initial_estimate


def optimizePoseGraph(graph, initial_estimate):
    parameters = gtsam.GaussNewtonParams()
    # Set optimization parameters
    parameters.setRelativeErrorTol(1e-5) # Stop when change in error is small
    parameters.setMaxIterations(100)     # Limit iterations
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, parameters)

    # Optimize!
    result = optimizer.optimize()
    marginals = gtsam.Marginals(graph, result)
    covariances = [marginals.marginalCovariance(i) for i in range(result.size())]
    return result, covariances


def incremental_solution_2d(poses, edges):
    isam = gtsam.ISAM2()
    result = None
    for pose in poses:
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        _, i, x, y, theta = pose
        if i == 0:
            graph.add(gtsam.PriorFactorPose2(i, Pose2(x, y, theta), gtsam.noiseModel.Diagonal.Variances(np.array([1e-6, 1e-6, 1e-8]))))
            initial_estimate.insert(i, Pose2(x, y, theta))
        else:
            prev_pose = poses[i-1]
            initial_estimate.insert(i, Pose2(prev_pose[2], prev_pose[3], prev_pose[4]))
        
        for edge in edges:
            _, ii, jj, dx, dy, dtheta, q = edge
            if  jj == i:

                pose_ = Pose2(dx, dy, dtheta)
                information_matrix = gtsam.noiseModel.Gaussian.Information(np.array([[q[0], q[1], q[2]],
                                                                                    [q[1], q[3], q[4]],
                                                                                    [q[2], q[4], q[5]]]))
                graph.add(gtsam.BetweenFactorPose2(ii, jj, pose_, information_matrix))
        isam.update(graph, initial_estimate)
        result = isam.calculateEstimate()
    return result


def showComparisonGraphs(poses1, poses2, title1="Initial Trajectory Estimate", title2="Optimized Trajectory Estimate", output_path=None):
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    plt.cla()

    # extract translation coordinates from both pose sets
    def extract_xy(poses):
        xs, ys = [], []
        for i in range(poses.size()):
            p = poses.atPose2(i)            
            xs.append(p.x()); ys.append(p.y())
        return xs, ys

    xs1, ys1 = extract_xy(poses1)
    xs2, ys2 = extract_xy(poses2)
    
    ax.plot(xs1, ys1, c='blue', label=title1, alpha=0.7, linewidth=0.75)
    sc2 = ax.plot(xs2, ys2, c='red', label=title2, alpha=0.7, linewidth=0.75)        
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    ax.set_title(f'{title1} vs {title2}')
    ax.legend()

    plt.savefig(f'pose2dImages/{output_path}.svg')


def showGraph(poses, cov=None, title="Initial Trajectory", output_path=None):    
    fig = plt.figure(0)
    axes = fig.gca()
    plt.cla()
    # Plot initial estimate poses
    for i in range(poses.size()):
        pose = poses.atPose2(i)
        cov_ = cov[i] if cov is not None else None      
        plot.plot_pose2(0, pose, axis_length=0.4, covariance = cov_) # Plotting with scale 0.5

    plt.axis('equal')    
    plt.title(title)
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(True)
    plt.savefig(f"pose2dImages/{output_path}.svg")
    plt.close()


def main():
    
    print("Leyendo datos...\n")
    vertexes, edges = readData('input_INTEL_g2o.g2o')

    print("Generando la estimacion inicial...\n")
    graph, initial_estimate = createPoseGraph(vertexes, edges)
    showGraph(initial_estimate, output_path='pose_graph_initial')
    marginals = gtsam.Marginals(graph, initial_estimate)
    showGraph(initial_estimate, cov=[marginals.marginalCovariance(i) for i in range(initial_estimate.size())],  output_path='pose_graph_initial_withCov')
    
    print("Optimizando el grafo de poses con Gauss Newton...\n")
    optimizedGN, covariances = optimizePoseGraph(graph, initial_estimate)
    showGraph(optimizedGN, title="Optimized Trajectory", output_path='pose_graph_optimized')
    showComparisonGraphs(initial_estimate, optimizedGN, output_path='comparison_initial_gn')

    print("Generando el grafo de poses de forma incremental...\n")
    incremental_result = incremental_solution_2d(vertexes, edges)
    showGraph(incremental_result, title="Incremental Optimized Trajectory", output_path='pose_graph_incremental_optimized')
    showComparisonGraphs(initial_estimate, incremental_result, title1="Initial Trajectory Estimate",title2="Incremental Trajectory Estimate", output_path='comparison_initial_incremental')

    
if __name__ == "__main__":
    main()
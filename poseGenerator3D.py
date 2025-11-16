import graphviz
import numpy as np
import plotly.graph_objects as go

import gtsam
from gtsam import Pose3
from gtsam.utils import plot
import matplotlib.pyplot as plt
import argparse

# Lee los datos del archivo g2o caso base lee parking-garage.g2o
def readData(file = 'parking-garage.g2o'):
    vertexes = []
    edges = []
    try:        
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
    except FileNotFoundError:
        print(f"Error: El archivo {file} no se encontró en el directorio 'data/'.")
    return vertexes, edges

# Crea la matriz de informacion 6x6 a partir de los valores leidos del archivo g2o
def create_information_matrix_3d(q):
    mgtsam = np.array([[q[0], q[1], q[2], q[3], q[4], q[5]],
                        [q[1], q[6], q[7], q[8], q[9], q[10]],
                        [q[2], q[7], q[11], q[12], q[13], q[14]],
                        [q[3], q[8], q[12], q[15], q[16], q[17]],
                        [q[4], q[9], q[13], q[16], q[18], q[19]],
                        [q[5], q[10], q[14], q[17], q[19], q[20]]])

    return gtsam.noiseModel.Gaussian.Information(mgtsam)

# Crea una Pose3 a partir de las traslaciones y cuaterniones
def create_pose3(dx, dy, dz, quaternions):
    rotation3 = gtsam.Rot3.Quaternion(quaternions[3], quaternions[0], quaternions[1], quaternions[2])
    point3 = gtsam.Point3(dx, dy, dz)
    pose3 = gtsam.Pose3(rotation3, point3)
    return pose3

# Estimacion inicial del grafo de poses 3D
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

# Perturba las estimaciones iniciales de las poses 3D
def pertubateEstimations(poses, sigma, seed):

    new_poses = gtsam.Values()

    np.random.seed(seed)

    # Perturbamos las poses
    for i in poses.keys():
        
        p_i = poses.atPose3(i)
        x_i = p_i.x() + np.random.normal(0, sigma[0])
        y_i = p_i.y() + np.random.normal(0, sigma[1])
        z_i = p_i.z() + np.random.normal(0, sigma[2])
        qx_i = p_i.rotation().toQuaternion().x() + np.random.normal(0, sigma[3])
        qy_i = p_i.rotation().toQuaternion().y() + np.random.normal(0, sigma[4])
        qz_i = p_i.rotation().toQuaternion().z() + np.random.normal(0, sigma[5])
        qw_i = p_i.rotation().toQuaternion().w() + np.random.normal(0, sigma[6])

        # Creamos la nueva pose perturbada
        new_pose = create_pose3(x_i, y_i, z_i, [qx_i, qy_i, qz_i, qw_i])

        new_poses.insert(i, new_pose)

    return new_poses

# Optimiza el grafo de poses 3D usando Gauss-Newton
def optimizePoseGraph(graph, initial_estimate):
    
    # Configuramos el optimizador Gauss-Newton
    parameters = gtsam.GaussNewtonParams()
    parameters.setAbsoluteErrorTol(1e-9) # Stop when change in error is small
    parameters.setRelativeErrorTol(1e-9) # Stop when change in error is small
    parameters.setMaxIterations(50)     # Limit iterations
    parameters.setVerbosity("Termination")    
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, parameters)

    # Optimizamos
    result = optimizer.optimize()    
    
    marginals = gtsam.Marginals(graph, result)
    covariances = [marginals.marginalCovariance(i) for i in range(result.size())]
    return result, covariances

# Solucion incremental del grafo de poses 3D
def incremental_solution_3d(poses, edges):
    isam = gtsam.ISAM2()
    result = None
    for pose in poses:
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        i, x, y, z, quaternions = pose
        pose3 = create_pose3(x, y, z, quaternions)
        if i == 0:
            priorModel = gtsam.noiseModel.Diagonal.Variances(np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]))
            graph.add(gtsam.PriorFactorPose3(i, pose3, priorModel))
            initial_estimate.insert(i, pose3)
        else:
            prev_pose = result.atPose3(i - 1)                
            initial_estimate.insert(i, prev_pose)
        
        for edge in edges:
            ii, jj,dx, dy, dz, quaternions, info = edge
            if  jj == i:

                pose_ = create_pose3(dx, dy, dz, quaternions)
                information_matrix = create_information_matrix_3d(info)
                graph.add(gtsam.BetweenFactorPose3(ii, jj, pose_, information_matrix))
        isam.update(graph, initial_estimate)
        result = isam.calculateEstimate()
    return result

# Muestra la comparacion entre dos grafos de poses 3D
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
    
    ax.plot(ys1, xs1, zs1, c='blue', label=title1, alpha=0.7, linewidth=0.75)
    sc2 = ax.plot(ys2, xs2, zs2, c='red', label=title2, alpha=0.7, linewidth=0.75)
    
    # shorten Z axis visually
    try:
        ax.set_box_aspect((1, 1, 0.2))  # (x, y, z) scale -> make z shorter
    except Exception:
        # fallback for older matplotlib: compress z-limits toward center
        zmin = min(min(zs1), min(zs2))
        zmax = max(max(zs1), max(zs2))
        zmid = 0.5 * (zmin + zmax)
        zhalf = 0.5 * (zmax - zmin) * 0.4
        if zhalf <= 0:
            zhalf = 1e-3
        ax.set_zlim(zmid - zhalf, zmid + zhalf)
    xmin = min(min(xs1), min(xs2)) * 1.2
    xmax = max(max(xs1), max(xs2)) * 2
    ymin = min(min(ys1), min(ys2)) * 1.2
    ymax = max(max(ys1), max(ys2)) * 1.2

    ax.set_xlim([ymax, ymin])
    ax.set_ylim([xmin, xmax])

    # rotate z-axis 45 degrees clockwise (negative azim rotates clockwise)
    ax.view_init(elev=40, azim=-25)
    ax.grid(False)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    ax.set_title(f'{title1} vs {title2}')
    ax.legend()

    plt.savefig(f'pose3dImages/{output_path}.svg')
    plt.close()
    
# Muestra el grafo de poses 3D
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
    plt.grid(False)
    plt.savefig(f"pose3dImages/{output_path}.svg")
    plt.close() 


def main(dataset='parking-garage.g2o'):
    
    # Ejercicio 3-A - Leyendo datos
    print("Leyendo datos...\n")
    vertexes, edges = readData(dataset)

    if not vertexes or not edges:
        print("No se pudieron leer los datos correctamente.")
        return
    
    # Ejercicio 3-B - Batch Solution
    print("Generando la estimacion inicial...\n")
    graph, initial_estimate = createPoseGraph3D(vertexes, edges)    
    showGraph3D(initial_estimate, title="Initial 3D Pose Graph", output_path="initial_3d_pose_graph")
    marginals = gtsam.Marginals(graph, initial_estimate)
    showGraph3D(initial_estimate, cov=[marginals.marginalCovariance(i) for i in range(initial_estimate.size())],  output_path='pose_graph_initial_withCov')

    print("Optimizando el grafo de poses con Gauss Newton...\n")
    optimizedGN, covariances = optimizePoseGraph(graph, initial_estimate)
    showGraph3D(optimizedGN, title="Optimized 3D Pose Graph", output_path="optimized_3d_pose_graph")
    showComparisonGraphs3D(initial_estimate, optimizedGN, title1="Unoptimized Trajectory", title2="Optimized Trajectory", output_path="3d_trajectory_comparison")

    print("Optimizando el grafo de poses con Gauss Newton perturbando la estimacion inicial...\n")
    perturbed_initial_estimate = pertubateEstimations(initial_estimate, (0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1), 0)
    optimizedGN, covariances = optimizePoseGraph(graph, perturbed_initial_estimate)
    showGraph3D(optimizedGN, title="Optimized 3D Pose Graph with Perturbation", output_path="optimized_3d_pose_graph_with_perturbation")
    showComparisonGraphs3D(initial_estimate, optimizedGN, title1="Unoptimized Trajectory", title2="Optimized Trajectory with Perturbation", output_path="3d_trajectory_comparison_with_perturbation")

    # Ejercicio 3-C - Incremental Solution
    print("Generando la solucion incremental...\n")
    incremental_result = incremental_solution_3d(vertexes, edges)
    showGraph3D(incremental_result, title="Incremental 3D Pose Graph", output_path="incremental_3d_pose_graph")
    showComparisonGraphs3D(initial_estimate, incremental_result, title1="Unoptimized Trajectory", title2="Incremental Trajectory", output_path="3d_incremental_trajectory_comparison")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesamiento de datos de un dataset G20 3d")
    parser.add_argument("--dataset", default='parking-garage.g2o', help="Dataset g2o 3d a procesar")
    args = parser.parse_args()
    main(dataset=args.dataset)  
# Trabajo Practico 4 Graph SLAM
FACULTAD DE CS. EXACTAS, INGENIERÍA Y AGRIMENSURA
LICENCIATURA EN CIENCIAS DE LA COMPUTACIÓN
ROBÓTICA MÓVIL

Autores: 
Franco Gozzerino
Jordi Solá

# Requerimientos

Instalar Anaconda3
Instalar GTSAM, mas explicitamente hablando, el wrapper de python de GTSAM. Para esto utilizaremos conda y necesitara descargarse su repositorio

```
conda create -n gtsam_env python=<your_python_version>
conda activate gtsam_env
conda install -c conda-forge cmake eigen pybind11 boost numpy matplotlib pythongraphviz pybind11-stubgen conda-forge::plotly conda-forge::pandas conda-forge::nbformat
```

```
git clone https://github.com/borglab/gtsam.git
cd gtsam
mkdir build && cd build
conda install -r <gtsam_folder>/python/requirements.txt
cmake .. -DGTSAM_BUILD_PYTHON=1 -DGTSAM_BUILD_UNSTABLE=OFF -DGTSAM_PYTHON_VERSION=<your_python_version>
make -j2 (2 is the number of threads you want to use)
export PYTHONPATH=$PYTHONPATH:$(pwd)/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
cd python
pip install .
cd ..
make python-install
```


Y una vez hecho se podra correr

# Uso

Este proyecto incluye 2 scripts:
- `poseGenerator.py`: que trabaja sobre datasets G2o 2D 
- `poseGenerator3D.py`: que trabaja sobre datasets G2o 3D 

Estos datasets deberan estar almacenados en la carpeta `data` y guardaran las imagenes resultantes en las carpetas `pose2dImages` y `pose3dImages` respectivamente


Donde ademas tendremos las siguientes opciones a la hora de correrlo:

- `--dataset <FILE>` El dataset a procesar

De no estar incluida esta bandera se utiliza el dataset 3d `parking-garage.g2o` o 2d `input_INTEL_g2o.g2o` correspondiente incluido en el trabajo

# Decisiones tomadas

A la hora de hacer la instalacion de GTSAM decidimos no instalar los paquetes de su version unstable debido a que no afectaban el desarrollo del trabajo, y sin embargo nos ocasionaba problemas a la hora de compilarlo. Este hecho esta ya incluido en la guia de instalación


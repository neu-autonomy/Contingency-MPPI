# Contingency-MPPI: Contingency Constrained Planning with MPPI within MPPI 

### **Accepted to L4DC 2025**

## Dependencies
Our code has been tested on Ubuntu 22.04. If you wish to run our code without use of Docker, please install the following dependencies:
```
sudo apt-get install libcdd-dev libblas3 libblas-dev liblapack3 liblapack-dev gfortran ccache libasio-dev 
```
If you also wish to run our code on hardware or Gazebo via ROS2-Humble:
```
sudo apt-get install ros-humble-gazebo-ros-pkgs ros-humble-navigation2
```
To use with SNOPT, request a trial license [here](https://ccom.ucsd.edu/~optimizers/licensing/). After receving the license file and libsnopt zip file, unzip the file, and
add both to your environment variables (it may be convenient to add these lines to a ./bashrc file):
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{path to unzipped libsnopt folder}
export SNOPT_LICENSE={path to license file}
```

## Setup (No ROS)
This is the setup to replicate our simulation results in our paper: 
1. Clone Repository
```
git clone https://github.com/neu-autonomy/Contingency-MPPI
cd Contingency-MPPI
```
2. Install requirements
```
pip3 install -r requirements.txt
# Optionally, if you wish to run with jax-cuda (recommended), run
# pip install -U "jax[cuda12]"
```
3. Install CMPPI
```
pip install -e .
```
## Setup (with ROS)
This is the setup to run our code within a gazebo environment (and with minimal changes, on hardware). As our tests are with the Agile-X Scout-Mini, we will install the tools needed to work with the Scout-Mini, as well as the Ros2-Nav2 stack and Gazebo
1. Clone Repository into a workspace
```
mkdir -p ws/src && cd ws/src
git clone https://github.com/neu-autonomy/Contingency-MPPI
```
2. Clone Scout/UGV packages
```
git clone https://github.com/westonrobot/ugv_sdk.git
git clone https://github.com/leonardjmihs/scout_ros2
```
3. Install the Nav2 stack/Gazebo
```
sudo apt-get install ros-humble-gazebo-ros-pkgs ros-humble-navigation2
```
4. Build
```
cd ..
colcon build
```
## Running Random Simulations
To recreate the sim results in our paper, run **nmppi_global_random.py** in '**branch_mppi/jax_mppi/**'
```
cd branch_mppi/jax_mppi
python nmppi_global_random.py --solver snopt
```
This will generate a random scenario and run each variant of our algorithms as well as the baseline algorithms, and save the results in a folder titled "**sim_results\*/**""
## Running Gazebo Simulation
To run a sample of our algorithm on Gazebo, first open 3 terminals, and run the following commands (don't forget to source the workspace):
1. ```ros2 launch scout_ugv_sim empty_world.launch.py```
2. ```ros2 launch branch_mppi nav2_costmap.launch.py```
3. ```ros2 launch branch_mppi branch_mppi_node_sim.launch.py```

You can visualize via rviz2: we provide a helpful rviz param file in **config/branch_mppi_view.rviz**
  ## Citation
When using CMPPI, please cite the following paper  ([pdf](https://arxiv.org/abs/2412.09777), [video](https://www.youtube.com/watch?v=RsXih-potZc))
```bibtex
@misc{jung2024contingencyconstrainedplanningmppi,
      title={Contingency Constrained Planning with MPPI within MPPI}, 
      author={Leonard Jung and Alexander Estornell and Michael Everett},
      year={2024},
      eprint={2412.09777},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2412.09777}, 
}
```

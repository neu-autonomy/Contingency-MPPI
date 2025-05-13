## Installation
- Create workspace and clone
  ```
  mkdir -p branch_mppi_ws/src
  cd branch_mppi_ws/src
  git clone https://github.com/leonardjmihs/mppi
  cd mppi
 
  ```
- Create virtual environment:

  ```
  python3 -m venv venv  --system-site-packages --symlinks
  source venv/bin/activate
  ```
- Install requirements
  ```
  pip install -r requirements.txt
  ```
- Build
  ```
  cd ../../
  colcon build
  # Note: due to virtual environments, --symlink-install doesn't really work 
  ```
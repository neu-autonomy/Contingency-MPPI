#!/usr/bin/env python
from setuptools import setup, find_packages
import os
from glob import glob
package_name = "branch_mppi"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
def _parse_requirements(file):
    with open(os.path.join(_CURRENT_DIR, file)) as f:
        return [line.rstrip() for line in f if not (line.isspace() or line.startswith("#"))]

packages_to_install = _parse_requirements("requirements.txt")
packages_to_install.append('setuptools')

setup(
    name=package_name,
    version="0.0.0",
    # packages=find_packages(package_name,exclude=["training","data","eval"]),
    # packages=find_packages("branch_mppi",exclude=['data','training']),
    packages=[package_name, 
            *(os.path.join(package_name,pkg) for pkg in find_packages(package_name, exclude=['data','training','experiments', 'controllers'])), 
            ],
    description="Playing around from MPPI (potentially with some neural-clbf stuff)",
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
],
    install_requires=packages_to_install,
    zip_safe=True,
    author="Leonard Jung",
    author_email="jung.le@northeastern.edu",
    tests_require=['pytest'],
    entry_points={
        "console_scripts": [
            "nested_mppi_node = branch_mppi.nested_mppi_node:main",
            "nested_mppi_one_plan = branch_mppi.nested_mppi_node_one_plan:main",
            "commander_node = branch_mppi.commander_node:main",
            "visualizer_node = branch_mppi.visualizer_node:main",
        ],
    }

)

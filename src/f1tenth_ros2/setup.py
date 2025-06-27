from setuptools import setup
import os 
from glob import glob
from setuptools import find_packages


package_name = 'f1tenth_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    py_modules=[
        '%s.scripts.joystick_ros2'%(package_name),
        # '%s.scripts.ftw_error'%(package_name),
        '%s.scripts.teleop_joy'%(package_name),
        '%s.scripts.waypoint_logger'%(package_name),
        # '%s.scripts.ftw_control'%(package_name),
        '%s.scripts.lookahead_point_viz'%(package_name),
        '%s.scripts.pp_controller'%(package_name),
        '%s.scripts.pure_pursuit'%(package_name),
        '%s.params.f110'%(package_name),
        '%s.mpc.Dynmpc_f1tenth'%(package_name),
        '%s.mpc.EKmpc_f1tenth'%(package_name),
        '%s.mpc.nmpc'%(package_name),
        '%s.mpc.ref_path_viz'%(package_name),
        # 'inputs'
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*_launch.py')),
        (os.path.join('share', package_name), glob('params/*.py')),
        (os.path.join('share', package_name), glob('models/*.py')),
        (os.path.join('share', package_name), glob('plots/*.py')),
        (os.path.join('share', package_name), glob('config/*.yaml')),
        (os.path.join('share', package_name), glob('mpc/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ning',
    maintainer_email='brianning1992@gmail.com',
    description='Autonomous racing used MPC in F1tenth gym simulator',
    license='Apache License, Version 2.0',
    keywords=['ROS'],
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joystick_ros2 = %s.scripts.joystick_ros2:main' %(package_name),
            # 'ftw_error = %s.scripts.ftw_error:main' %(package_name),
            'teleop_joy = %s.scripts.teleop_joy:main' %(package_name),
            'waypoint_logger = %s.scripts.waypoint_logger:main' %(package_name),
            # 'ftw_control = %s.scripts.ftw_control:main' %(package_name),
            'lookahead_point = %s.scripts.lookahead_point_viz:main' %(package_name),
            'pp_controller = %s.scripts.pp_controller:main' %(package_name),
            'mpc_controller = %s.mpc.Dynmpc_f1tenth:main' %(package_name),
            'reference_path = %s.mpc.ref_path_viz:main' %(package_name),
            'gpmpc_controller = %s.mpc.EKmpc_f1tenth:main' %(package_name),
        ],
    },
)
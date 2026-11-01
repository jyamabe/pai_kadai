from glob import glob
import os

from setuptools import setup

package_name = 'my_robot_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'teleop_keyboard = my_robot_teleop.teleop_keyboard:main',
        ],
    },
)

from setuptools import find_packages, setup

package_name = 'sorting_robot_task'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'task_planner = '
            'sorting_robot_task.task_planner:main',
            'manipulation_sequence = '
            'sorting_robot_task.manipulation_sequence:main',
            'task_planner2 = '
            'sorting_robot_task.task_planner2:main',
        ],
    },
)

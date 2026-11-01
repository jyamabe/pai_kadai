from setuptools import find_packages, setup


package_name = 'sorting_robot_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(
        exclude=['test']
    ),
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
    install_requires=[
        'setuptools'
    ],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@example.com',
    description='Color detection for sorting robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
    #        'color_detector = '
    #        'sorting_robot_perception.color_detector:main',
            'ai_detector = '
            'sorting_robot_perception.ai_detector:main',

            'imitation_policy = '
            'sorting_robot_perception.imitation_policy_node:main',
        ],
    },
)

from setuptools import find_packages, setup

package_name = 'coord_light_yolov5'

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
    maintainer='long',
    maintainer_email='thientran2032@outlook.com',
    description='ZED YOLO package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_detection_node = scripts.coord_light_yolov5_inference:main',
        ],
    },
)

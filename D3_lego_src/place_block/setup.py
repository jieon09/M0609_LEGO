from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'place_block'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'resource'),
            glob('resource/*')
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seongho',
    maintainer_email='20201124@edu.hanbat.ac.kr',
    description='Doosan M0609 block picker using 24x24 grid JSON and calibration file',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'place_block16 = place_block.place_block_node:main',
            'place_block_ori = place_block.block_pick_node_original:main',

            
        
        ],
    },
)

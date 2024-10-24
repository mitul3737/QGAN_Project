from setuptools import setup, find_packages

setup(
    name='QGAN_Project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'pennylane',
        'pennylane-qchem'
    ],
    entry_points={
        'console_scripts': [
            'qgan_train=qgan.train:main'
        ]
    },
)

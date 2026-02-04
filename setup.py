from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='lung_ultrasound',
    version='0.1',
    description='A package for lung ultrasound image analysis',
    author='Angelo Lasala',
    author_email='angelo.lasala@predictcare.it',
    packages=find_packages(),
    install_requires=requirements,
)
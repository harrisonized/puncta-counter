from setuptools import setup, find_packages

with open('./README.md') as f:
    description = f.read()

setup(
    name='puncta-counter',
    description='Module for quantifying CellProfiler output tables',
    long_description=description,
    version='1.0',
    author='Harrison Wang',
    author_email='harrison.c.wang@gmail.com',
    url='https://github.com/harrisonized/puncta-counter',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
)

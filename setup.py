from setuptools import setup

setup(
    name='pytorch_utils',
    version='0.1.0',
    author='Young Mo Kang',
    author_email='kang.youngmo@gmail.com',
    packages=['logger'],
    scripts=[],
    description='Useful utilities for training with Pytorch.',
    install_requires=[
        "torch >= 0.4",
        "matplotlib >= 3.0",
        "numpy >= 1.15",
    ],
)
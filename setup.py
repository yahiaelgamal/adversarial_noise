from setuptools import setup, find_packages

# Read the requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='adversial_noise',
    version='1.0',
    description='A Python package for adding adversial noise to images using PyTorch.',
    author='Yahia Elgamal',
    author_email='yahiaelgamal@gmail.com',
    # packages=find_packages(),
    packages=find_packages('src'),  # Look for packages in the 'src' directory

    # package_dir={'': '/src'},
    install_requires=requirements,
)

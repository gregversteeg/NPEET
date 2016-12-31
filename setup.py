from setuptools import setup, find_packages

required = []

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='npeet',
      version='1.0.0',
      description='Non-parametric Entropy Estimation Toolbox',
      author='Greg Ver Steeg',
      author_email='gregv@isi.edu',
      url='https://github.com/MaxwellRebo/NPEET',
      packages=find_packages(),
      install_requires=required)

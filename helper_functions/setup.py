from setuptools import setup

setup(
      name='zangorth-ramsey',
      version='1.1.0',
      description='Helper Functions for Ramsey Project',
      author='Zangorth',
      packages=['ramsey'],
      install_requires=open(r'C:\Users\Samuel\Google Drive\Portfolio\Ramsey\helper_functions\requirements.txt', 'r').read().split('\n')[0:-1]
      )
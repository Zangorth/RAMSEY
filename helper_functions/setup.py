import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE / 'README.md').read_text()
REQS = (HERE / 'requirements.txt').read_text()


setup(
      name='zangorth-ramsey',
      version='1.1.5',
      description='Helper Functions for Ramsey Project',
      author='Zangorth',
      packages=['ramsey'],
      install_requires=REQS.split('\n')[0:-1]
      )
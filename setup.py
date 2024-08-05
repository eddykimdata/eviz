from setuptools import setup, find_packages

setup(
  name='eviz',
  version='0.1.1',
  author='E K',
  author_email='jettabebetta@gmail.com',
  packages=['eviz'],
  # scripts=['bin/script1','bin/script2'],
  # url='http://pypi.python.org/pypi/PackageName/',
  license='LICENSE.txt',
  description='Enhanced Matplotlib visualizations',
  long_description=open('README.md').read(),
  install_requires=[
    'pandas==2.2.2',
    'numpy==2.0.1',
    'matplotlib==3.9.0',
    'seaborn==0.13.2'
  ],
)
from setuptools import setup, find_packages

setup(
  name='eviz',
  version='0.1.0',
  author='E K',
  author_email='jettabebetta@gmail.com',
  packages=['eviz'],
  # scripts=['bin/script1','bin/script2'],
  # url='http://pypi.python.org/pypi/PackageName/',
  license='LICENSE.txt',
  description='Enhanced Matplotlib visualizations',
  long_description=open('README.md').read(),
  install_requires=[
    'pandas==1.3.5',
    'numpy==1.21.6',
    'matplotlib==3.5.1',
    'seaborn==0.12.1'
  ],
)
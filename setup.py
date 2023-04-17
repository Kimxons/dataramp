from setuptools import setup
from setuptools import find_packages
import io
from os.path import join
from os.path import dirname

def read(*names, **kwargs):
    try:
        with io.open(
                .join(dirname(__file__), *names),
                encoding=kwargs.get("encoding", "utf8")
                ) as fr:
            return fr.read()
        except:
            pass

'''
with open("README.md", "r") as md:
    long_description = md.read()

with open("requirements.txt") as f:
    requirements = f.readlines()
'''

setup(
        name='data_help',
        verion='0.1.0',
        license='MIT',
        description='A python library for easy modelling',
        long_description=read("README.md"),
        long_description_content_type='text/markdown',
        author='shaks',
        author_email='dev.kitonga@gmail.com',
        url='https://github.com',
        packages=find_packages("data_help"),
        package_dir={"":"data_help"},
        install_requires=[
            'pandas',
            'scikit-learn',
            'seaborn',
            'numpy',
            'jupyter',
            'matplotlib',
            'nltk',
            'joblib',
            ],
        classifiers=[
            'Development Status :: 1 - Development/Unstable',
            'Intended Audience :: Developers',
            'Programming Language :: Python :: 3.10.6',
            ],
     )

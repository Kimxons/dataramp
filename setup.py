from setuptools import setup, find_packages

with open("README.md", "r") as md:
    long_description = md.read()

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
        name='data_help',
        verion='1.0.0',
        license='MIT',
        description='A python library for easy modelling',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='shaks',
        author_email='dev.kitonga@gmail.com',
        url='https://github.com',
        packages=find_packages(),
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

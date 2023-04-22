from setuptools import setup, find_packages
import io
from os.path import join, dirname


def read(*names, **kwargs):
    try:
        with io.open(join(dirname(__file__), *names),
                     encoding=kwargs.get("encoding", "utf8")) as fr:
            return fr.read()
    except (IOError, UnicodeError) as e:
        print(f"Error reading the file: {e}")
        return None


setup(
    name='data_help',
    version='0.1.0',
    license='MIT',
    description='A python library for easy modelling',
    long_description=read("README.md"),
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
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.10',
    ],
)

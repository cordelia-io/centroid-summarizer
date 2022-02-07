from setuptools import setup, find_packages


with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name="centroid_summarizer",
    version="0.1",
    url="https://github.com/cordelia-io/centroid-summarizer",
    author="Jakub Bartczuk, austinjp",
    maintainer="austinjp",
    packages=find_packages(),
    install_requires=requirements
)

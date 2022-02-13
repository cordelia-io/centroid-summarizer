import pathlib
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent
readme = (here / "README.md").read_text()

with open(here / "requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name=find_packages()[0],
    description="Extractive text summarization using centroid distance",
    long_description = readme,
    long_description_content_type = "text/markdown",
    license = "GPL-3.0-or-later",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Text Processing"
    ],
    version="0.0.1",
    url="https://github.com/cordelia-io/centroid-summarizer",
    author="Gaetano Rossiello, Pierpaolo Basile, Giovanni Semeraro, Jakub Bartczuk, olivier-compilatio, Vinicius Monteiro, austinjp",
    maintainer="austinjp",
    packages=find_packages(),
    install_requires=requirements
)

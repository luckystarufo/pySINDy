<p align="left">
  <a href="https://pysindy.readthedocs.io/en/latest/" target="_blank" >
    <img alt="Python Sparse Identification of Nonelinear Dynamics" src="docs/source/_static/logo_PySINDy.png" width="250" />
  </a>
</p>

</p>
<p align="left">
    <a href="#travis" alt="Travis Build Status">
        <img src="https://travis-ci.org/luckystarufo/pySINDy.svg?branch=master" /></a>
    <a href="#docs" alt="Documentation Status">
        <img src="https://readthedocs.org/projects/pysindy/badge/?version=latest" /></a>
</p>   

**pySINDy**: python Sparse Identification of Nonlinear Dynamics

## Table of contents
* [Introduction](#introduction)
* [Structure](#structure)
* [Getting Started](#getting-started)
    * [Prerequisite](#prerequisite)
    * [Installing](#installing)
* [Usage](#usage)
    * [Examples](#examples)
    * [Running the tests](#running-the-tests)
* [License](#license)
* [Acknowledgments](#acknowledgments)

## Introduction
PySINDy is a Python package that implements the SINDy-family algorithms. 
SINDy is short for "Sparse Identification of Nonlinear Dynamics", which 
is a class of data-driven algorithms for system identification. This class 
of algorithms are mainly developed by Steve Brunton and Nathan Kutz at the 
University of Washington. Since then, many variants arose, such as SINDy for 
PDEs, implicit SINDy, parametric SINDy, Hybrid SINDy, SINDy with control, etc. 
In PySINDy, we will (or we will try our best to) implement the majority of 
the variants with a friendly user interface, see the [Examples](#examples) 
section for more details. 

The idea behind the SINDy algorithm is not terribly new. To simply put it, 
it just automatically calculate some spatial and temporal derivatives from 
some high-fidelity measurements data and does a sparse regression of some sort.
In other words, you feed it with some time series measurements, then it provides
you a differential equation model with sparse coefficients. (Why sparse? It is 
just an assumption on 'parsimony'!)

One last note: From our experience, the algorithms do rely on high-fidelity measurements
so that it calculates the right 'derivatives'. If you have nosiy data, please preprocess 
first and then use the algorithm, or you can try to calculate the derivatives with some 
interpolation methods to reduce the noise effects. We may also add some features on that 
later.

## Structure
    pySINDy/
      |- README.md
      |- datasets
         |- burgers.mat
         |- reaction_diffusion.mat
         |- ...
      |- env
         |- ...
      |- pySINDy/
         |- __init__.py
         |- sindybase.py
         |- sindy.py
         |- isindy.py
         |- sindypde.py
         |- data/
            |- ...
         |- tests/
            |- ...
         |- utils/
            |- generator.py
      |- examples
         |- example-1-sindy-vanderpol.ipynb
         |- example-2-sindypde-burgers.ipynb
         |- example-3-sindypde-reactiondiffusion.ipynb
         |- example-4-isindy-xxx.ipynb
      |- docs/
         |- Design Doc.pdf
         |- Makefile
         |- make.bat
         |- build/
            |- ...
         |- source/
            |- _static
            |- _summaries
            |- conf.py
            |- index.rst
            |- ...
      |- setup.py
      |- .gitignore
      |- .travis.yml
      |- LICENSE
      |- requirements.txt


## Getting Started

PySINDy requires numpy, scipy, matplotlib, findiff, pytest (for unittests), pylint (for PEP8 style check), 
sphinx (for documentation). The code is compatible with Python 3.5 and Python 3.6. It can be installed 
using pip or directly from the source code.

### Installing via pip

Mac and Linux users can install pre-built binary packages using pip. To install the package just type:
```bash
> pip install pysindy
```

### Installing from source

The official distribution is on GitHub, and you can clone the repository using:
```bash
> git clone https://github.com/luckystarufo/pySINDy
```
Then, to install the package just type:
```bash
> python setup.py install
```

## Usage
**PySINDy** uses [Sphinx](http://www.sphinx-doc.org/en/stable/) for code documentation.
So you can see more details about the code usage [there](https://pysindy.readthedocs.io/en/latest/).

### Examples
We will frequently update simple examples for demo purposes, and here are currently exisiting
ones:
1. [SINDy: Van Der Pol Oscillator](examples/example-1-sindy-vanderpol.ipynb)
2. [SINDyPDE: Burgers Equation](examples/example-2-sindypde-burgers.ipynb)
3. [SINDyPDE: Reaction Diffusion](examples/example-3-sindypde-reactiondiffusion.ipynb)
4. [ISINDy example](examples/example-4-isindy-xxx.ipynb)

### Running the tests
We are using Travis CI for continuous intergration testing. You can check out the current status 
[here](https://travis-ci.org/luckystarufo/pySINDy).

To run tests locally, type:
```bash
> pytest pySINDy
```

## License
This project utilizes the [MIT LICENSE](LICENSE).
100% open-source, feel free to utilize the code however you like. 

## Acknowledgments
PySINDy is primarily developed for CSE 583 - Software Development for Data Scientist
at the University of Washington, so special thanks to the instructors David A. C. Beck,
Joseph L. Hellerstein, Bernease Herman and Colin Lockard. 
And of course, special thanks to two other contributors (my teammates: Yi Chu and Lianzong Wang), 
who contributed a lot in implementing some of the algorithms, performing unittests as well as benchmarking.

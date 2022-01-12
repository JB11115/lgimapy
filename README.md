# lgimapy

![Python Versions][python-versions]
![BSD License][license]

This is a general purpose python library for LGIM America with modules for
for loading/interacting with data from various databases, modeling/building
indexes, performing quantitative analysis, and publishing reports.


## Table of contents
* [Documentation](#documentation)
* [Setup](#setup)
* [Authors](#authors)

## Setup
Install this project locally by running:

```bash
git clone https://gogs.btprod.openshift.inv.adroot.lgim.com/JB11115/lgimapy.git
```

To set up a local environment using Anaconda:

```bash
cd lgimapy/
conda env create -f envs/lgimapy38_base_env.yml
source activate lgimapy
pip install pypdf4
pip install -e .
```
\* Note that `pip install` may fail if behind proxy or firewall. In that case
use: `pip install --trusted-host files.pythonhosted.org --trusted-host pypi.org --trusted-host pypi.python.org pypdf4`

#### Other Requirements
* pdflatex: installed with apt package `texlive`
* various latex libraries: installed with apt packages `texlive-latex-extra` and
  `texlive-fonts-extra`
* `bbgpy`, a Bloomberg API python package, installed in Windows 10
* A logged-in Bloomberg terminal in Windows 10

## Documentation
To build documentation, from the root directory run:
```bash
cd docs
make html
cd build/html
open index.html
```
This URL will not change, so it is recommended to bookmark it in
the browser of your choice. Documentation can be updated at any time
using the `make html` command.

## Authors

The main developer(s):

- Jason Becker: jason.becker@lgima.com

[python-versions]: https://img.shields.io/badge/python-3.8-blue.svg
[license]: https://img.shields.io/badge/license-TBD-green

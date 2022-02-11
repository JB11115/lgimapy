# lgimapy

![Python Versions][python-versions]

This is a general purpose python library for LGIM America with modules for
for loading/interacting with data from various databases, scraping Bloomberg
data, performing quantitative analysis, constructing portfolio trades,
plotting figures, and publishing reports.


## Table of contents
* [Setup](#setup)
* [Documentation](#documentation)
* [Authors](#authors)

## Setup

#### Code

Install this project locally by running:

```bash
git clone https://gogs.btprod.openshift.inv.adroot.lgim.com/JB11115/lgimapy.git
```

To set up a local environment using Anaconda:

```bash
cd lgimapy/
conda env create -f envs/lgimapy38_base_env.yml
conda activate lgimapy38
pip install pypdf4
pip install -e .
```
\* Note that `pip install` may fail if behind proxy or firewall. In that case
use: `pip install --trusted-host files.pythonhosted.org --trusted-host pypi.org --trusted-host pypi.python.org pypdf4`

#### Data
To copy over the current data files, run the following. Note that this will
take an hour or more to complete.

```bash
rsync -rv /mnt/x/Credit\ Strategy/lgimapy/data/ data/
```


#### Other Requirements
* pdflatex: used for building reports, installed with `apt` package `texlive`
* various latex libraries: used building reports, installed with `apt` packages
`texlive-latex-extra` and `texlive-fonts-extra`
* A logged-in Bloomberg terminal in Windows 10 for scraping Bloomberg data
* Python installed in Windows, with `pandas`, `blpapi`, and `pybbg` libraries.
* [Bloomberg API] C/C++ Supported Release v3.17.1.1
* [Visual Studio C++ Build Tools]

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

[Bloomberg API]: https://www.bloomberg.com/professional/support/api-library/
[Visual Studio C++ Build Tools]: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=2ahUKEwiniM2o26HlAhVQRKwKHS_8DrAQFjAAegQIABAB&url=https%3A%2F%2Fgo.microsoft.com%2Ffwlink%2F%3FLinkId%3D691126&usg=AOvVaw0geDw_h-TSCfzTMvYE2ZOw
[python-versions]: https://img.shields.io/badge/python-3.8-blue.svg

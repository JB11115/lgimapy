# lgimapy

![Python Versions][python-versions]
![BSD License][license]

This is a general purpose python library for LGIMA
with modules for loading/interacting with data, modeling/building indexes,
performing quantitative analysis, and publishing reports.


## Table of contents
* [Documentation](#documentation)
* [Setup](#setup)
* [Authors](#authors)


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


## Setup
Install this project locally using:n

```bash
git clone https://gogs.btprod.openshift.inv.adroot.lgim.com/JB11115/lgimapy.git
```

To set up a local environment using Anaconda:

```bash
cd lgimapy/
conda create -n lgimapy python=3.6
source activate lgimapy
pip install -U pip
pip install -U setuptools
pip install -r requirements.txt
pip install git+https://github.com/kyuni22/pybbg  # Bloomberg API
conda install -c conda-forge blpapi  # Bloomberg API
pip install -e .
```
\* Note that `pip install` may fail if behind proxy or firewall. Set up
`.bashrc` shell aliases as shown below and replace with `pipinstall` to get
around this issue.

#### Other Requirements
* [Bloomberg API] C/C++ Supported Release v3.12.3.1
* [Visual Studio C++ Build Tools] (for convex optimization)
* [Microsoft Build Tools] (for convex optimization)
* MikTex, download from the Software Center (for building pdf documents)

#### Setting Environment Variables
Using Windows start menu, search for `environment` and select
`Edit environment variables for your account`.
Ensure the following Variable:Value pairs exist in User variables
(verify install locations and versions where necessary):
* `PYTHONPATH`: `C:\LGIM\Conda\python.exe`
* `BLPAPI_ROOT`: `C:\BLP\BloombergWindowsSDK\BloombergWindowsSDK\C++API\v3.12.2.1`
* `path`: `C:\blp\DAPI;C:\blp\DAPI\DDE;C:\BLP\BloombergWindowsSDK\BloombergWindowsSDK\C++API\v3.12.2.1;C:\LGIM\Conda\python.exe;C:\LGIM\Conda;C:\LGIM\Conda\Scripts;C:\Program Files (x86)\MiKTeX 2.9\miktex\bin\`

#### Setting Shell Path Variables
In your home directory create `.bash_profile` and `.bashrc` files:
```bash
cd ~
touch .bash_profile
touch .bashrc
```

Structure your `.bash_profile` as below replacing `USERNAME` and `PASSWORD` with
your login username and password:
```bash
source ~/.bashrc;

export http_proxy=http://USERNAME:PASSWORD@proxych:8080

export https_proxy=$http_proxy
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$http_proxy
export ftp_proxy=$http_proxy
export rsync_proxy=$http_proxy
export no_proxy="localhost,127.0.0.1,localaddress,.yourcompany.com,.local"
```

Structure your `.bashrc` as below
(verify install locations and versions where necessary):
```bash
# Path variables
export PATH=$PATH:"/C/blp/DAPI"
export PATH=$PATH:"/C/blp/DAPI/DDE"
export PATH=$PATH:"/C/LGIM/Conda"
export PATH=$PATH:"/C/LGIM/Conda/python.exe"
export PATH=$PATH:"/C/LGIM/Conda/Scripts"
export PATH=$PATH:"/C/Program Files (x86)/MiKteX 2.9/miktex/bin"
export PATH=$PATH:"/C/BLP/BloombergWindowsSDK/BloombergWindowsSDK/C++API/v3.12.2.1"
export BLPAPI_ROOT=$PATH:"/C/BLP/BloombergWindowsSDK/BloombergWindowsSDK/C++API/v3.12.2.1"

# pip install through proxy and firewall
alias pipinstall='pip install --trusted-host files.pythonhosted.org --trusted-host pypi.org --trusted-host pypi.python.org'
```
## Authors

The main developer(s):

- Jason R Becker: jason.becker@lgima.com

[Bloomberg API]: https://www.bloomberg.com/professional/support/api-library/
[Visual Studio C++ Build Tools]: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=2ahUKEwiniM2o26HlAhVQRKwKHS_8DrAQFjAAegQIABAB&url=https%3A%2F%2Fgo.microsoft.com%2Ffwlink%2F%3FLinkId%3D691126&usg=AOvVaw0geDw_h-TSCfzTMvYE2ZOw
[Microsoft Build Tools]: https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16

[python-versions]: https://img.shields.io/badge/python-3.6-blue.svg
[license]: https://img.shields.io/badge/license-TBD-green

lgimapy
=======

|python-Versions|

``lgimapy`` is a general purpose python library for LGIMA
 with modules for modeling/building indexes, modeling treasury curves,
 performing analysis, and misc. utility functions.

Documentation: \\inv\lgima\data\Credit Strategy\lgimapy\docs\build\html\index.html

TO VIEW THIS README AS INTENDED USE THE FOLLOWING COMMAND FROM TERMINAL:

.. code:: sh

   restview README.rst


.. contents:: Table of contents
   :backlinks: top
   :local:

Installation
------------

Install Repo
~~~~~~~~~~~~


From terminal:

.. code:: sh

   git clone https://gogs.btprod.openshift.inv.adroot.lgim.com/JB11115/lgimapy.git


Set up venv
~~~~~~~~~~~

Using Anaconda, from terminal:

.. code:: sh

   cd lgimapy/
   conda create -n lgimapy python=3.6
   source activate lgimapy
   pip install -U pip
   pip install -r requirements.txt
   conda install rpy2



LICENSE
-------

TBD


Authors
-------

The main developer(s):

- Jason R Becker (`jrbecker <https://github.com/jason-r-becker>`__)


.. |Python-Versions| image:: https://img.shields.io/badge/python-3.6-blue.svg

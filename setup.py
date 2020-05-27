from glob import glob
from os.path import abspath, basename, dirname, join, splitext

from setuptools import find_packages, setup

# Get the long description from the README file
with open(join(abspath(dirname(__file__)), 'README.md')) as f:
    long_description = f.read()

setup(
    name='lgimapy',
    version='0.0.1',
    license='TBD',
    description='General purpose library for LGIMA.',
    long_description=long_description,
    author='Jason R. Becker',
    author_email='jason.becker@lgima.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: TBD',
        'Operating System :: Microsoft :: Windows :: Windows 7'
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: Anaconda',
        'Topic :: Office/Business :: Financial',
        'Topic :: Office/Business :: Financial :: Investment',
        ],
    keywords=['LGIMA', 'Index', 'Curve', 'Modeling'],
    )

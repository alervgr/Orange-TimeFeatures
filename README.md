Orange3 TimeFeatures
===============

Timefeatures add-on for [Orange] 3 data mining software for generating synthetic data using datasets with time series, generating graphs of relationships between the generated variables and includes another widget to save the data and configuration tables in a database.

[Orange]: https://orangedatamining.com/

Installation
------------

### Orange add-on installer

Install from Orange add-on installer through Options -> Add-ons.

![Installation](https://github.com/alervgr/Orange-TimeFeatures/blob/main/imgs/installation.png?raw=true)

### Using pip

To install the add-on with pip use

    pip install TimeFeatures

To install the add-on from source, run

    python setup.py install

To register this add-on with Orange, but keep the code in the development directory (do not copy it to 
Python's site-packages directory), run

    python setup.py develop

You can also run

    pip install -e .

which is sometimes preferable as you can *pip uninstall* packages later.

### Anaconda

If using Anaconda Python distribution, simply run

    pip install TimeFeatures

**Required Dependencies**:
* numpy>=1.22.4
* AnyQt>=0.2.0
* Orange3>=3.34.0
* PyQt5>=5.15.6
* scipy>=1.7.3
* Orange3-Network>=1.8.0

Usage
-----

After the installation, the widgets from this add-on are registered with Orange. To run Orange from the terminal,
use

    orange-canvas

or

    python3 -m Orange.canvas

New widgets are in the toolbox bar under Time-Features section.

Workflow Example
-----
This is an example of how you can use this add-on.

![Workflow](https://github.com/alervgr/Orange-TimeFeatures/blob/main/imgs/workflow.png?raw=true)

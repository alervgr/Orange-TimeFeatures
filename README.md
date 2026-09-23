# Orange3 TimeFeatures

[![PyPI version](https://img.shields.io/pypi/v/TimeFeatures)](https://pypi.org/project/TimeFeatures/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/alervgr/Orange-TimeFeatures/actions/workflows/ci.yml/badge.svg)](https://github.com/alervgr/Orange-TimeFeatures/actions/workflows/ci.yml)
[![Orange3](https://img.shields.io/badge/Orange3-add--on-orange)](https://orangedatamining.com/)

TimeFeatures is an add-on for [Orange] 3 data mining software for generating synthetic data using datasets with time series, generating graphs of relationships between the generated variables, and includes widgets to save and load data and configuration tables from a database.

[Orange]: https://orangedatamining.com/

## Features

- 🕐 **7 time-window functions** — `shift`, `sum`, `mean`, `min`, `max`, `count`, `sd` with full chunk-boundary correctness
- 🔗 **Chained descriptors** — derived variables can reference each other; topological sort resolves the evaluation order automatically
- 🧮 **Python-style expressions** — evaluated with a restricted `eval` (`__builtins__` replaced, curated whitelist only). This is **not** a security sandbox: expressions can still run arbitrary code, so only open workflows from sources you trust
- 🗄️ **PostgreSQL & MySQL** — persist and reload datasets via SQLAlchemy with dialect-agnostic SQL generation
- 📊 **Directed weighted dependency graphs** — edge weights reflect temporal window size; visualise in Network Explorer
- ⚡ **Bulk upload performance** — pandas `DataFrame.to_sql` with chunked multi-row INSERTs
- 💾 **Workflow persistence** — variable definitions survive save/reload without clicking Send first

## Widgets

| Widget | Description |
|---|---|
| **Time Features Constructor** | Defines new variables from existing ones using Python-style expressions and time-window functions. Supports chained descriptors with automatic topological sorting. |
| **Variable Dependency Graph** | Builds a directed, weighted dependency graph from the variable definitions. Edge weights summarise how far back or forward in time each variable looks. |
| **Save to DB** | Persists the resulting dataset to a SQL database (PostgreSQL or MySQL), with full SQL-injection defences and three write modes (create / overwrite / append). |
| **Load from DB** | Lists datasets previously stored by Save to DB and pulls the chosen one back into Orange, optionally marking the class column directly so no Select Columns widget is needed. |

## Installation

### Orange add-on installer

Install from Orange add-on installer through Options -> Add-ons.

![Installation](https://github.com/alervgr/Orange-TimeFeatures/blob/main/imgs/installation.png?raw=true)

### Using pip

To install the add-on with pip use

    pip install TimeFeatures

To install the add-on from source in editable mode, run

    pip install -e .

### Anaconda

If using Anaconda Python distribution, simply run

    pip install TimeFeatures

**Required Dependencies** (Python 3.10 or newer):

* Orange3>=3.36.2
* numpy>=1.22.4
* AnyQt>=0.2.0
* PyQt5>=5.15.6
* PyQtWebEngine>=5.15.6
* scipy>=1.7.3
* SQLAlchemy>=1.4.0
* psycopg2-binary>=2.9.9
* PyMySQL>=1.0.0
* Orange3-Network>=1.8.0

## Usage

After the installation, the widgets from this add-on are registered with Orange. To run Orange from the terminal,
use

    orange-canvas

or

    python3 -m Orange.canvas

New widgets are in the toolbox bar under Time-Features section.

## Documentation

The add-on includes Sphinx documentation for each widget. Orange resolves the
local HTML pages through its internal Help panel, not through an internet URL.
To rebuild the documentation locally, run

    pip install -e ".[docs]"
    python -m sphinx -b html docs docs/build/html

The bundled in-app help is pre-built under `timefeatures/help_html/`. To
regenerate it (e.g. after editing the `.rst` files), run

    python -m sphinx -b html docs timefeatures/help_html

Use the widget help action in Orange to open the corresponding page inside the
Orange Help window.

## Workflow Example

This is an example of how you can use this add-on.

![Workflow](https://github.com/alervgr/Orange-TimeFeatures/blob/main/imgs/workflow.png?raw=true)

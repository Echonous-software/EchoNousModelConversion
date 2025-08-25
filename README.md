# EchoNous Model Conversion

One stop shop for pytorch to mobile runtime model conversion.

## Installation

Create a virtual environment (or use an existing) one. I'm using python 3.12
as it has some improvements to static type hinting and pattern matching
which I'd like to use, but the project may be usable on older python versions
as well.
```commandline
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Recommend editable installation (`-e` flag), which means changes in the src
directory are automatically applied. On need to re-run the install command
if new dependencies are added or other changes to the build system itself.

```commandline
pip install -e ".[dev]"
```

At least once, run the sync-ml-projects command:

```commandline
python scripts/sync-ml-projects.py
```

## Usage

So far, run the loaders.py file to load and print info about
all the models we have integrated so far:

```commandline
python -m echonous.models.loaders
```
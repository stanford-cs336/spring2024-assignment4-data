# CS336 Spring 2024 Assignment 4: Data

For a full description of the assignment, see the assignment handout at
[cs336_spring2024_assignment4_data.pdf](./cs336_spring2024_assignment4_data.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `setup.py`. This module should contain your
  from-scratch language model from assignment 1. At this point, **you are free
  to optimize this model however you wish**---feel free to replace your
  hand-crafted components with PyTorch built-ins wherever you like.
- [`./cs336-data`](./cs336-data): directory containing a module
  `cs336_data` and its associated `setup.py`. In this module, you will
  implement code to filter and preprocess data.

Visually, it should look something like:

``` sh
.
├── cs336-basics # Files from assignment 1, feel free to make optimizations 
│   ├── cs336_basics # A python module named cs336_basics
│   │   ├── __init__.py
│   │   ├── VERSION
│   │   └── ... other files in the cs336_basics module, taken from assignment 1 ...
│   ├── requirements.txt
│   └── setup.py (setup.py to install `cs336_basics`) 
├── cs336-data # TODO(you):code that you'll write for assignment 4 
│   ├── cs336_data # A python module named cs336_data
│   │   ├── __init__.py
│   │   ├── VERSION
│   │   └── ... TODO(you): other python files that you need for assignment 4 ...
│   ├── requirements.txt
│   ├── ... TODO(you): any other files or folders you need for assignment 4 ...
│   └── setup.py (setup.py to install `cs336_data`)
├── README.md
└── ... TODO(you): other files or folders you need for assignment 4 ...
```

1. Set up a conda environment and install packages. In particular, the
   `cs336-basics` package (located at [`./cs336-basics`](./cs336-basics))
   installs the `cs336_basics` module, and the `cs336-data` package (located
   at [`./cs336-data`](./cs336-data)) installs the `cs336_data` module.

``` sh
conda create -n cs336_data python=3.10 --yes
conda activate cs336_data
pip install -e ./cs336-basics/ -e ./cs336-data/'[test]'
```

2. Activate the environment:

``` sh
conda activate cs336_data
```

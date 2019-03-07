[![PyPI Shield](https://img.shields.io/pypi/v/wind.svg)](https://pypi.python.org/pypi/wind)
[![Travis CI Shield](https://travis-ci.org/D3-AI/wind.svg?branch=master)](https://travis-ci.org/D3-AI/wind)

# Wind

<p align="center">
<i>Machine learning for internet of things related to wind systems.  </I>
</p>




<p align="center">
<i>A collaborative open source project between Data to AI Lab at MIT and Xylem Inc. </I>
</p>


- Free software: MIT license
- Documentation: https://D3-AI.github.io/wind
- Homepage: https://github.com/D3-AI/wind

# Overview

The Wind project is a collection of end-to-end solutions for machine learning tasks commonly
found in monitoring wind energy production systems. Most tasks utilize sensor data
emanating from monitoring systems. We utilize the foundational innovations developed for
automation of machine Learning at Data to AI Lab at MIT. This project is developed in close
collaboration with Iberdrola, S.A.

The salient aspects of this customized project are:
* A set of ready to use, well tested pipelines for different machine learning tasks. These are
  vetted through testing across multiple publicly available datasets for the same task.
* An easy interface to specify the task, pipeline, and generate results and summarize them.
* A production ready, deployable pipeline.
* An easy interface to ``tune`` pipelines using Bayesian Tuning and Bandits library.
* A community oriented infrastructure to incorporate new pipelines.
* A robust continuous integration and testing infrastructure.
* A ``learning database`` recording all past outcomes --> tasks, pipelines, outcomes.

## Concepts

Before diving into the software usage, we briefly explain some concepts and terminology.

### Primitive

We call the smallest computational blocks used in a Machine Learning process
**primitives**, which:

* Can be either classes or functions.
* Have some initialization arguments, which MLBlocks calls `init_params`.
* Have some tunable hyperparameters, which have types and a list or range of valid values.

### Template

Primitives can be combined to form what we call **Templates**, which:

* Have a list of primitives.
* Have some initialization arguments, which correspond to the initialization arguments
  of their primitives.
* Have some tunable hyperparameters, which correspond to the tunable hyperparameters
  of their primitives.

### Pipeline

Templates can be used to build **Pipelines** by taking and fixing a set of valid
hyperparameters for a Template. Hence, Pipelines:

* Have a list of primitives, which corresponds to the list of primitives of their template.
* Have some initialization arguments, which correspond to the initialization arguments
  of their template.
* Have some hyperparameter values, which fall within the ranges of valid tunable
  hyperparameters of their template.

A pipeline can be fitted and evaluated using the MLPipeline API in MLBlocks.


## Current tasks and pipelines

In our current phase, we are addressing two tasks - time series classification and time series
regression. To provide solutions for these two tasks we have two components.

### WindPipeline

This class is the one in charge of learning from the data and making predictions by building
[MLBlocks](https://hdi-project.github.io/MLBlocks) and later on tuning them using
[BTB](https://hdi-project.github.io/BTB/)

This class comes in two flavours in the form of subclasses, the **WindClassifier** and the
**WindRegressor**, to be used in the corresponding problem types.

### WindLoader

A class responsible for loading the time series data from CSV files, and return it in the
format ready to be used by the **WindPipeline**.

### Wind Dataset

A dataset is a folder that contains time series data and information about
a Machine Learning problem in the form of CSV and JSON files.

The expected contents of the `dataset` folder are:

* A `metadata.json` with information about all the tables found in the dataset.
  This file follows the [Metadata.json schema](https://github.com/HDI-Project/MetaData.json)
  with three small modifications:
  * The root document has a `name` entry, with the name of the dataset.
  * The foreign key columns are be of type `id` and subtype `foreign`.
  * The `datetime` columns that are time indexes need to have the `time_index` subtype.

* A CSV file containing the training samples with, at least, the following columns:
  * A unique index
  * A foreign key to at least one timeseries table
  * A time index that works as the cutoff time for the training example
  * If the problem is supervised, a target column.

Then, for each type of timeseries that exist in the dataset, there will be:

* A CSV file containing the id of each timeseries and any additional information associated with it
* A CSV file containing the timeseries data with the following columns:
  * A unique index
  * A foreign key to the timeseries table
  * A time index
  * At least a value column

### Tuning

We call tuning the process of, given a dataset and a template, find the pipeline derived from the
given template that gets the best possible score on the given dataset.

This process usually involves fitting and evaluating multiple pipelines with different hyperparameter
values on the same data while using optimization algorithms to deduce which hyperparameters are more
likely to get the best results in the next iterations.

We call each one of these tries a **tuning iteration**.


# Getting Started

## Installation

The simplest and recommended way to install **Wind** is using pip:

```bash
pip install wind
```

For development, you can also clone the repository and install it from sources

```bash
git clone git@github.com:D3-AI/wind.git
cd wind
make install-develop
```

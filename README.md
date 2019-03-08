
[![PyPI Shield](https://img.shields.io/pypi/v/wind.svg)](https://pypi.python.org/pypi/wind)
[![Travis CI Shield](https://travis-ci.org/D3-AI/wind.svg?branch=master)](https://travis-ci.org/D3-AI/wind)

# Wind

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

### WindLoader

A class responsible for loading the time series data from CSV files, and return it in the
format ready to be used by the **WindPipeline**.


### Wind Dataset

A dataset is a folder that contains time series data and information about
a Machine Learning problem in the form of CSV and JSON files.

The expected contents of the `dataset` folder are 4 CSV files:

* A **Turbines** table that contains:
  * `turbine_id`: column with the unique id of each turbine.
  * A number of additional columns with information about each turbine.
* A **Signals** table that contains:
  * `signal_id`: column with the unique id of each signal.
  * A number of additional columns with information about each signal.
* A **Readings** table that contains:
  * `reading_id`: Unique identifier of this reading.
  * `turbine_id`: Unique identifier of the turbine which this reading comes from.
  * `signal_id`: Unique identifier of the signal which this reading comes from.
  * `timestamp`: Time where the reading took place, as an ISO formatted datetime.
  * `value`: Numeric value of this reading.
* A **Targets** table that contains:
  * `target_id`: Unique identifier of the turbine which this label corresponds to.
  * `turbine_id`: Unique identifier of the turbine which this label corresponds to.
  * `timestamp`: Time associated with this target
  * `target`: The value that we want to predict. This can either be a numerical value or a categorical label.

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

## Usage Example

In this example we will load some demo data using the **WindLoader** and fetch it to the
**WindPipeline** for it to find the best possible pipeline, fit it using the given data
and then make predictions from it.

### Load and explore the data

We first create a loader instance passing:

* The path to the dataset folder
* The name of the target table
* The name of the target column
* Optionally, the names of the readings, turbines and signals tables, in case they are different from the default ones.


```python
from wind.loader import WindLoader

loader = WindLoader('examples/datasets/wind/', 'labels', 'label')
```

Then we call the `loader.load` method, which will return three elements:

* `X`: The contents of the target table, where the training examples can be found, without the target column.
* `y`: The target column, as exctracted from the the target table.
* `tables`: A dictionary containing the additional tables that the Pipeline will need to run, `readings`, `turbines` and `signals`.


```python
X, y, tables = loader.load()
X.head(5)
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label_id</th>
      <th>turbine_id</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>2013-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>2013-01-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>2013-01-03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>2013-01-04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>2013-01-05</td>
    </tr>
  </tbody>
</table>
</div>




```python
y.head(5)
```




    0    0.0
    1    0.0
    2    0.0
    3    0.0
    4    0.0
    Name: label, dtype: float64




```python
tables.keys()
```




    dict_keys(['readings', 'signals', 'turbines'])




```python
tables['turbines'].head()
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>turbine_id</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Turbine 0</td>
    </tr>
  </tbody>
</table>
</div>




```python
tables['signals'].head()
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>signal_id</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>WTG01_Grid Production PossiblePower Avg. (1)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>WTG02_Grid Production PossiblePower Avg. (2)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>WTG03_Grid Production PossiblePower Avg. (3)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>WTG04_Grid Production PossiblePower Avg. (4)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>WTG05_Grid Production PossiblePower Avg. (5)</td>
    </tr>
  </tbody>
</table>
</div>




```python
tables['readings'].head()
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reading_id</th>
      <th>turbine_id</th>
      <th>signal_id</th>
      <th>timestamp</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2013-01-01</td>
      <td>817.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2013-01-01</td>
      <td>805.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>786.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>2013-01-01</td>
      <td>809.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>2013-01-01</td>
      <td>755.0</td>
    </tr>
  </tbody>
</table>
</div>



### Split the data

If we want to split the data in train and test subsets, we can do so by splitting the `X` and `y` variables.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```

### Finding the best Pipeline

Once we have loaded the data, we create a **WindPipeline** instance by passing:

* `template (string)`: the name of a template or the path to a template json file.
* `metric (string or function)`: The name of the metric to use or a metric function to use.
* `cost (bool)`: Whether the metric is a cost function to be minimized or a score to be maximized.

Optionally, we can also pass defails about the cross validation configuration:
* `stratify`
* `cv_splits`
* `shuffle`
* `random_state`


```python
from wind.pipeline import WindPipeline

pipeline = WindPipeline('wind_classification', 'accuracy', cv_splits=2)
```

    Using TensorFlow backend.


Once we have created the pipeline, we can call its `tune` method to find the best possible
hyperparameters for our data, passing the `X`, `y`, and `tables` variables returned by the loader,
as well as an indication of the number of tuning iterations that we want to perform.


```python
pipeline.tune(X_train, y_train, tables, iterations=0)
```

After the tuning process has finished, the hyperparameters have been already set in the classifier.

We can see the found hyperparameters by calling the `get_hyperparameters` method.


```python
import json

print(json.dumps(pipeline.get_hyperparameters(), indent=4))
```

    {
        "pandas.DataFrame.resample#1": {
            "rule": "1D",
            "time_index": "timestamp",
            "groupby": [
                "turbine_id",
                "signal_id"
            ],
            "aggregation": "mean"
        },
        "pandas.DataFrame.unstack#1": {
            "level": "signal_id",
            "reset_index": true
        },
        "featuretools.EntitySet.entity_from_dataframe#1": {
            "entityset_id": "entityset",
            "entity_id": "readings",
            "index": "index",
            "variable_types": null,
            "make_index": true,
            "time_index": "timestamp",
            "secondary_time_index": null,
            "already_sorted": false
        },
        "featuretools.EntitySet.entity_from_dataframe#2": {
            "entityset_id": "entityset",
            "entity_id": "turbines",
            "index": "turbine_id",
            "variable_types": null,
            "make_index": false,
            "time_index": null,
            "secondary_time_index": null,
            "already_sorted": false
        },
        "featuretools.EntitySet.entity_from_dataframe#3": {
            "entityset_id": "entityset",
            "entity_id": "signals",
            "index": "signal_id",
            "variable_types": null,
            "make_index": false,
            "time_index": null,
            "secondary_time_index": null,
            "already_sorted": false
        },
        "featuretools.EntitySet.add_relationship#1": {
            "parent": "turbines",
            "parent_column": "turbine_id",
            "child": "readings",
            "child_column": "turbine_id"
        },
        "featuretools.dfs#1": {
            "target_entity": "turbines",
            "index": "turbine_id",
            "time_index": "timestamp",
            "agg_primitives": null,
            "trans_primitives": null,
            "copy": false,
            "encode": false,
            "max_depth": 1,
            "remove_low_information": true
        },
        "mlprimitives.custom.feature_extraction.CategoricalEncoder#1": {
            "copy": true,
            "features": "auto",
            "max_labels": 0
        },
        "sklearn.impute.SimpleImputer#1": {
            "missing_values": NaN,
            "fill_value": null,
            "verbose": false,
            "copy": true,
            "strategy": "mean"
        },
        "sklearn.preprocessing.StandardScaler#1": {
            "with_mean": true,
            "with_std": true
        },
        "xgboost.XGBClassifier#1": {
            "n_jobs": -1,
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "gamma": 0,
            "min_child_weight": 1
        }
    }


as well as the obtained cross validation score by looking at the `score` attribute of the `tsc` object


```python
pipeline.score
```




    0.6592421640188922



Once we are satisfied with the obtained cross validation score, we can proceed to call
the `fit` method passing again the same data elements.


```python
pipeline.fit(X_train, y_train, tables)
```

After this, we are ready to make predictions on new data


```python
predictions = pipeline.predict(X_test, tables)
predictions[0:5]
```




    array([0., 0., 0., 0., 0.])



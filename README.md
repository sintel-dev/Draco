<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="DAI" />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<p align="left">
<img width=20% src="https://dai.lids.mit.edu/wp-content/uploads/2019/03/GreenGuard.png" alt="Draco" />
</p>

<p align="left">
AutoML for Renewable Energy Industries.
</p>


[![PyPI Shield](https://img.shields.io/pypi/v/draco-ml.svg)](https://pypi.python.org/pypi/draco-ml)
[![Tests](https://github.com/sintel-dev/Draco/workflows/Run%20Tests/badge.svg)](https://github.com/sintel-dev/Draco/actions?query=workflow%3A%22Run+Tests%22+branch%3Amaster)
[![Downloads](https://pepy.tech/badge/draco-ml)](https://pepy.tech/project/draco-ml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sintel-dev/Draco/master?filepath=tutorials)
<!--
[![Coverage Status](https://codecov.io/gh/sintel-dev/Draco/branch/master/graph/badge.svg)](https://codecov.io/gh/sintel-dev/Draco)
-->

# Draco

- License: [MIT](https://github.com/sintel-dev/Draco/blob/master/LICENSE)
- Documentation: https://sintel-dev.github.io/Draco
- Homepage: https://github.com/sintel-dev/Draco

## Overview

The Draco project is a collection of end-to-end solutions for machine learning problems
commonly found in monitoring wind energy production systems. Most tasks utilize sensor data
emanating from monitoring systems. We utilize the foundational innovations developed for
automation of machine Learning at Data to AI Lab at MIT.

The salient aspects of this customized project are:

* A set of ready to use, well tested pipelines for different machine learning tasks. These are
  vetted through testing across multiple publicly available datasets for the same task.
* An easy interface to specify the task, pipeline, and generate results and summarize them.
* A production ready, deployable pipeline.
* An easy interface to ``tune`` pipelines using Bayesian Tuning and Bandits library.
* A community oriented infrastructure to incorporate new pipelines.
* A robust continuous integration and testing infrastructure.
* A ``learning database`` recording all past outcomes --> tasks, pipelines, outcomes.

## Resources

* [Data Format](DATA_FORMAT.md).
* [Draco folder structure](DATA_FORMAT.md#folder-structure).

# Install

## Requirements

**Draco** has been developed and runs on Python 3.6, 3.7 and 3.8.

Also, although it is not strictly required, the usage of a [virtualenv](
https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid interfering
with other software installed in the system where you are trying to run **Draco**.

## Download and Install

**Draco** can be installed locally using [pip](https://pip.pypa.io/en/stable/) with
the following command:

```bash
pip install draco-ml
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

If you want to install from source or contribute to the project please read the
[Contributing Guide](https://sintel-dev.github.io/Draco/contributing.html#get-started).

# Data Format

The minimum input expected by the **Draco** system consists of the following two elements,
which need to be passed as `pandas.DataFrame` objects:

## Target Times

A table containing the specification of the problem that we are solving, which has three
columns:

* `turbine_id`: Unique identifier of the turbine which this label corresponds to.
* `cutoff_time`: Time associated with this target
* `target`: The value that we want to predict. This can either be a numerical value or a
  categorical label. This column can also be skipped when preparing data that will be used
  only to make predictions and not to fit any pipeline.

|    | turbine_id   | cutoff_time         |   target |
|----|--------------|---------------------|----------|
|  0 | T1           | 2001-01-02 00:00:00 |        0 |
|  1 | T1           | 2001-01-03 00:00:00 |        1 |
|  2 | T2           | 2001-01-04 00:00:00 |        0 |

## Readings

A table containing the signal data from the different sensors, with the following columns:

  * `turbine_id`: Unique identifier of the turbine which this reading comes from.
  * `signal_id`: Unique identifier of the signal which this reading comes from.
  * `timestamp (datetime)`: Time where the reading took place, as a datetime.
  * `value (float)`: Numeric value of this reading.

|    | turbine_id   | signal_id   | timestamp           |   value |
|----|--------------|-------------|---------------------|---------|
|  0 | T1           | S1          | 2001-01-01 00:00:00 |       1 |
|  1 | T1           | S1          | 2001-01-01 12:00:00 |       2 |
|  2 | T1           | S1          | 2001-01-02 00:00:00 |       3 |
|  3 | T1           | S1          | 2001-01-02 12:00:00 |       4 |
|  4 | T1           | S1          | 2001-01-03 00:00:00 |       5 |
|  5 | T1           | S1          | 2001-01-03 12:00:00 |       6 |
|  6 | T1           | S2          | 2001-01-01 00:00:00 |       7 |
|  7 | T1           | S2          | 2001-01-01 12:00:00 |       8 |
|  8 | T1           | S2          | 2001-01-02 00:00:00 |       9 |
|  9 | T1           | S2          | 2001-01-02 12:00:00 |      10 |
| 10 | T1           | S2          | 2001-01-03 00:00:00 |      11 |
| 11 | T1           | S2          | 2001-01-03 12:00:00 |      12 |

## Turbines

Optionally, a third table can be added containing metadata about the turbines.
The only requirement for this table is to have a `turbine_id` field, and it can have
an arbitraty number of additional fields.

|    | turbine_id   | manufacturer   | ...   | ...   | ...   |
|----|--------------|----------------|-------|-------|-------|
|  0 | T1           | Siemens        | ...   | ...   | ...   |
|  1 | T2           | Siemens        | ...   | ...   | ...   |

## CSV Format

A part from the in-memory data format explained above, which is limited by the memory
allocation capabilities of the system where it is run, **Draco** is also prepared to
load and work with data stored as a collection of CSV files, drastically increasing the amount
of data which it can work with. Further details about this format can be found in the
[project documentation site](DATA_FORMAT.md#csv-format).

# Quickstart

In this example we will load some demo data and classify it using a **Draco Pipeline**.

## 1. Load and split the demo data

The first step is to load the demo data.

For this, we will import and call the `draco.demo.load_demo` function without any arguments:

```python3
from draco.demo import load_demo

target_times, readings = load_demo()
```

The returned objects are:

*  ``target_times``: A ``pandas.DataFrame`` with the ``target_times`` table data:

   ```
     turbine_id cutoff_time  target
   0       T001  2013-01-12       0
   1       T001  2013-01-13       0
   2       T001  2013-01-14       0
   3       T001  2013-01-15       1
   4       T001  2013-01-16       0
   ```

* ``readings``: A ``pandas.DataFrame`` containing the time series data in the format explained above.

   ```
     turbine_id signal_id  timestamp  value
   0       T001       S01 2013-01-10  323.0
   1       T001       S02 2013-01-10  320.0
   2       T001       S03 2013-01-10  284.0
   3       T001       S04 2013-01-10  348.0
   4       T001       S05 2013-01-10  273.0
   ```

Once we have loaded the `target_times` and before proceeding to training any Machine Learning
Pipeline, we will have split them in 2 partitions for training and testing.

In this case, we will split them using the [train_test_split function from scikit-learn](
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html),
but it can be done with any other suitable tool.

```python3
from sklearn.model_selection import train_test_split

train, test = train_test_split(target_times, test_size=0.25, random_state=0)
```

Notice how we are only splitting the `target_times` data and not the `readings`.
This is because the pipelines will later on take care of selecting the parts of the
`readings` table needed for the training based on the information found inside
the `train` and `test` inputs.

Additionally, if we want to calculate a goodness-of-fit score later on, we can separate the
testing target values from the `test` table by popping them from it:

```python3
test_targets = test.pop('target')
```

## 2. Exploring the available Pipelines

Once we have the data ready, we need to find a suitable pipeline.

The list of available Draco Pipelines can be obtained using the `draco.get_pipelines`
function.

```python3
from draco import get_pipelines

pipelines = get_pipelines()
```

The returned `pipeline` variable will be `list` containing the names of all the pipelines
available in the Draco system:

```
['dfs_xgb',
 'dfs_xgb_with_unstack',
 'dfs_xgb_with_normalization',
 'dfs_xgb_with_unstack_normalization',
 'dfs_xgb_prob_with_unstack_normalization']
```

For the rest of this tutorial, we will select and use the pipeline
`dfs_xgb_with_unstack_normalization` as our template.

```python3
pipeline_name = 'dfs_xgb_with_unstack_normalization'
```

## 3. Fitting the Pipeline

Once we have loaded the data and selected the pipeline that we will use, we have to
fit it.

For this, we will create an instance of a `DracoPipeline` object passing the name
of the pipeline that we want to use:

```python3
from draco.pipeline import DracoPipeline

pipeline = DracoPipeline(pipeline_name)
```

And then we can directly fit it to our data by calling its `fit` method and passing in the
training `target_times` and the complete `readings` table:

```python3
pipeline.fit(train, readings)
```

## 4. Make predictions

After fitting the pipeline, we are ready to make predictions on new data by calling the
`pipeline.predict` method passing the testing `target_times` and, again, the complete
`readings` table.

```python3
predictions = pipeline.predict(test, readings)
```

## 5. Evaluate the goodness-of-fit

Finally, after making predictions we can evaluate how good the prediction was
using any suitable metric.

```python3
from sklearn.metrics import f1_score

f1_score(test_targets, predictions)
```

## What's next?

For more details about **Draco** and all its possibilities and features, please check the
[project documentation site](https://sintel-dev.github.io/Draco/)
Also do not forget to have a look at the [tutorials](
https://github.com/sintel-dev/Draco/tree/master/tutorials)!

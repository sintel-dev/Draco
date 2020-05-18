<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="DAI" />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<p align="left">
<img width=20% src="https://dai.lids.mit.edu/wp-content/uploads/2019/03/GreenGuard.png" alt="GreenGuard" />
</p>

<p align="left">
AutoML for Renewable Energy Industries.
</p>


[![PyPI Shield](https://img.shields.io/pypi/v/greenguard.svg)](https://pypi.python.org/pypi/greenguard)
[![Travis CI Shield](https://travis-ci.org/signals-dev/GreenGuard.svg?branch=master)](https://travis-ci.org/signals-dev/GreenGuard)
[![Downloads](https://pepy.tech/badge/greenguard)](https://pepy.tech/project/greenguard)
<!--
[![Coverage Status](https://codecov.io/gh/signals-dev/GreenGuard/branch/master/graph/badge.svg)](https://codecov.io/gh/signals-dev/GreenGuard)
-->

# GreenGuard

- License: [MIT](https://github.com/signals-dev/GreenGuard/blob/master/LICENSE)
- Documentation: https://signals-dev.github.io/GreenGuard
- Homepage: https://github.com/signals-dev/GreenGuard

# Overview

The GreenGuard project is a collection of end-to-end solutions for machine learning problems
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

# Install

## Requirements

**GreenGuard** has been developed and runs on Python 3.6 and 3.7.

Also, although it is not strictly required, the usage of a [virtualenv](
https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid interfering
with other software installed in the system where you are trying to run **GreenGuard**.

## Download and Install

**GreenGuard** can be installed locally using [pip](https://pip.pypa.io/en/stable/) with
the following command:

```bash
pip install greenguard
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

If you want to install from source or contribute to the project please read the
[Contributing Guide](https://d3-ai.github.io/GreenGuard/contributing.html#get-started).

## Docker usage

Alternatively, **GreenGuard** is prepared to be run inside a docker environment using
`docker-compose`.

For this, make sure to have both [docker](https://docs.docker.com/install/) and [docker-compose](
https://docs.docker.com/compose/install/) installed on your system and then follow these steps:

1. Clone this repository and go into the `GreenGuard` folder:

```bash
git clone git@github.com:signals-dev/GreenGuard.git
cd GreenGuard
```

2. Start a Jupyter Notebook inside a docker container.

```bash
docker-compose up --build
```

3. Point your browser at http://127.0.0.1:8888

# Data Format

The minimum input expected by the **GreenGuard** system consists of the following two elements,
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
allocation capabilities of the system where it is run, **GreenGuard** is also prepared to
load and work with data stored as a collection of CSV files, drastically increasing the amount
of data which it can work with. Further details about this format can be found in the
[project documentation site](https://d3-ai.github.io/GreenGuard/advanced_usage/csv.html).

# Quickstart

In this example we will load some demo data and classify it using a **GreenGuard Pipeline**.

## 1. Load and split the demo data

The first step is to load the demo data.

For this, we will import and call the `greenguard.demo.load_demo` function without any arguments:

```python3
from greenguard.demo import load_demo

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

The list of available GreenGuard Pipelines can be obtained using the `greenguard.get_pipelines`
function.

```python3
from greenguard import get_pipelines

pipelines = get_pipelines()
```

The returned `pipeline` variable will be `list` containing the names of all the pipelines
available in the GreenGuard system:

```
['resample_600s_normalize_dfs_1d_xgb_classifier',
 'resample_600s_unstack_normalize_dfs_1d_xgb_classifier',
 'resample_600s_unstack_double_144_lstm_timeseries_classifier',
 'resample_3600s_unstack_24_lstm_timeseries_classifier',
 'resample_3600s_unstack_double_24_lstm_timeseries_classifier',
 'resample_600s_unstack_dfs_1d_xgb_classifier',
 'resample_600s_unstack_144_lstm_timeseries_classifier']
```

For the rest of this tutorial, we will select and use the pipeline
`resample_600s_unstack_normalize_dfs_1d_xgb_classifier` as our template.

```python3
pipeline_name = 'resample_600s_unstack_normalize_dfs_1d_xgb_classifier'
```

## 3. Fitting the Pipeline

Once we have loaded the data and selected the pipeline that we will use, we have to
fit it.

For this, we will create an instance of a `GreenGuardPipeline` object passing the name
of the pipeline that we want to use:

```python3
from greenguard.pipeline import GreenGuardPipeline

pipeline = GreenGuardPipeline(pipeline_name)
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

For more details about **GreenGuard** and all its possibilities and features, please check the
[project documentation site](https://signals-dev.github.io/GreenGuard/)
Also do not forget to have a look at the [notebook tutorials](
https://github.com/signals-dev/GreenGuard/tree/master/notebooks)!

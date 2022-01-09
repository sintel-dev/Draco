# History

## 0.1.0 - 2022-01-01

* First release on ``draco-ml`` PyPI

## Previous GreenGuard development

### 0.3.0 - 2021-01-22

This release increases the supported version of python to `3.8` and also includes changes
in the installation requirements, where ``pandas`` and ``scikit-optimize`` packages have
been updated to support higher versions. This changes come together with the newer versions
of ``MLBlocks`` and ``MLPrimitives``.

#### Internal Improvements

* Fix ``run_benchmark`` generating properly the ``init_hyperparameters`` for the pipelines.
* New ``FPR`` metric.
* New ``roc_auc_score`` metric.
* Multiple benchmarking metrics allowed.
* Multiple ``tpr`` or ``threshold`` values allowed for the benchmark.

### 0.2.6 - 2020-10-23

* Fix ``mkdir`` when exporting to ``csv`` file the benchmark results.
* Intermediate steps for the pipelines with demo notebooks for each pipeline.

#### Resolved Issues

* Issue #50: Expose partial outputs and executions in the ``GreenGuardPipeline``.

### 0.2.5 - 2020-10-09

With this release we include:

* `run_benchmark`: A function within the module `benchmark` that allows the user to evaluate
templates against problems with different window size and resample rules.
* `summarize_results`: A function that given a `csv` file generates a `xlsx` file with a summary
tab and a detailed tab with the results from `run_benchmark`.

### 0.2.4 - 2020-09-25

* Fix dependency errors

### 0.2.3 - 2020-08-10

* Added benchmarking module.

### 0.2.2 - 2020-07-10

#### Internal Improvements

* Added github actions.

#### Resolved Issues

* Issue #27: Cache Splits pre-processed data on disk

### 0.2.1 - 2020-06-16

With this release we give the possibility to the user to specify more than one template when
creating a GreenGuardPipeline. When the `tune` method of this is called, an instance of BTBSession
is returned and it is in charge of selecting the templates and tuning their hyperparameters until
achieving the best pipeline.

#### Internal Improvements

* Resample by filename inside the `CSVLoader` to avoid oversampling of data that will not be used.
* Select targets now allows them to be equal.
* Fixed the csv filename format.
* Upgraded to BTB.

#### Bug Fixes

* Issue #33: Wrong default datetime format

#### Resolved Issues

* Issue #35: Select targets is too strict
* Issue #36: resample by filename inside csvloader
* Issue #39: Upgrade BTB
* Issue #41: Fix CSV filename format

### 0.2.0 - 2020-02-14

First stable release:

* efficient data loading and preprocessing
* initial collection of dfs and lstm based pipelines
* optimized pipeline tuning
* documentation and tutorials

### 0.1.0

* First release on PyPI

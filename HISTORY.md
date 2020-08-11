# History

## 0.2.3 - 2020-08-10

* Added benchmarking module.

## 0.2.2 - 2020-07-10

### Internal Imrpovements

* Added github actions.

### Resolved Issues

* Issue #27: Cache Splits pre-processed data on disk

## 0.2.1 - 2020-06-16

With this release we give the possibility to the user to specify more than one template when
creating a GreenGuardPipeline. When the `tune` method of this is called, an instance of BTBSession
is returned and it is in charge of selecting the templates and tuning their hyperparameters until
achieving the best pipeline.

### Internal Improvements

* Resample by filename inside the `CSVLoader` to avoid oversampling of data that will not be used.
* Select targets now allows them to be equal.
* Fixed the csv filename format.
* Upgraded to BTB.

### Bug Fixes

* Issue #33: Wrong default datetime format

### Resolved Issues

* Issue #35: Select targets is too strict
* Issue #36: resample by filename inside csvloader
* Issue #39: Upgrade BTB
* Issue #41: Fix CSV filename format

## 0.2.0 - 2020-02-14

First stable release:

* efficient data loading and preprocessing
* initial collection of dfs and lstm based pipelines
* optimized pipeline tuning
* documentation and tutorials

## 0.1.0

* First release on PyPI

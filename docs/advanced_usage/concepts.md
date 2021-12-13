# Concepts

Here we briefly explain some of the concepts and terminology used within the Draco
project and documentation.

## Primitive

We call the smallest computational blocks used in a Machine Learning process
**primitives**, which:

* Can be either classes or functions.
* Have some initialization arguments, which MLBlocks calls `init_params`.
* Have some tunable hyperparameters, which have types and a list or range of valid values.

## Template

Primitives can be combined to form what we call **Templates**, which:

* Have a list of primitives.
* Have some initialization arguments, which correspond to the initialization arguments
  of their primitives.
* Have some tunable hyperparameters, which correspond to the tunable hyperparameters
  of their primitives.

## Pipeline

Templates can be used to build **Pipelines** by taking and fixing a set of valid
hyperparameters for a Template. Hence, Pipelines:

* Have a list of primitives, which corresponds to the list of primitives of their template.
* Have some initialization arguments, which correspond to the initialization arguments
  of their template.
* Have some hyperparameter values, which fall within the ranges of valid tunable
  hyperparameters of their template.

A pipeline can be fitted and evaluated directly using [MLBlocks](
https://MLBazaar.github.io/MLBlocks), or using the **DracoPipeline**.

## Tuning

We call tuning the process of, given a dataset and a collection of templates, finding the pipeline
derived from the templates that gets the best possible score on the dataset.

This process usually involves fitting and evaluating multiple pipelines with different
hyperparameter configurations on the same data while using optimization algorithms to deduce
which hyperparameters are more likely to get the best results in the next iterations.

We call each one of these evaluations a **tuning iteration**.

The process of selecting and tuning the templates is handled by a [BTBSession](
https://MLBazaar.github.io/BTB/tutorials/03_Session.html), which is responsible for
discarding the templates that do not work on the given data and for keeping
track of the template and hyperparameters that obtain the best performance.

## DracoPipeline

This class is the one in charge of loading the **MLBlocks Pipelines** configured in the
system and use them to learn from the data and make predictions.

This class is also responsible for creating the BTBSession that will handle the
selection and tuning of the templates.

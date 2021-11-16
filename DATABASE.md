# Database Schema

The **Draco Database** contains the following collections and relationships

* Farm
* Trubine
    * Farm
* Signal
* Sensor
    * Turbine
    * Signal
* Reading
    * Sensor
* PipelineTemplate
* Pipeline
    * PipelineTemplate
* MLTask
    * Turbine - multiple
* Target
    * MLTask
* Experiment
    * MLTask
    * PipelineTemplate
    * Signal - multiple
* ExperimenRun
    * Experiment
* PipelineRun
    * Pipeline
    * ExperimentRun

## Farm

A **Farm** represents a physical Wind Turbines Farm. This collection groups together multiple
Turbines with shared properties, such as location.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Object
* `name (String)`: Name or code given to this Object
* `insert_time (DateTime)`: Time when this Object was inserted
* `created_by (String)`: Identifier of the user that created this Object

## Turbine

A **Turbine** represents a physical Turbine. A Turbine is part of a **Farm**, and has some
particular properties, such as the Turbine manufacturer.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Object
* `farm_id (ObjectID)`: Unique Identifier of the Farm to which this Turbine belongs
* `name (String)`: Name or code given to this Object
* `manufacturer (String)`: Name or code of the manufacturer - EXAMPLE
* `model (String)`: Name or code of the model - EXAMPLE
* `insert_time (DateTime)`: Time when this Object was inserted
* `created_by (String)`: Identifier of the user that created this Object

## Signal

The **Signal** collection contains the details about each Signal type.
This includes shared properties of the signal, such as the sensor type or the measurement units.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Object
* `name (String)`: Name or code given to this Object
* `type (String)`: Type of sensor - EXAMPLE
* `created_by (String)`: Identifier of the user that created this Object
* `insert_time (DateTime)`: Time when this Object was inserted

## Sensor

A **Sensor** represents a physical sensor that is installed in a Turbine.
The Sensor collection specifies the turbine and the signal type, as well as properties
about the Sensor such as the Sensor manufacturer and model and its age.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Object
* `turbine_id (ObjectID)`: Unique Identifier of the Turbine where this Sensor is installed
* `signal_id (ObjectID)`: Unique Identifier of the Signal type of this Sensor
* `name (String)`: Name or code given to this Object
* `manufacturer (String)`: Name or code of the manufacturer - EXAMPLE
* `model (String)`: Name or code of the model - EXAMPLE
* `installation_date (DateTime)`: Time when this Sensor was installed - EXAMPLE
* `insert_time (DateTime)`: Time when this Object was inserted
* `created_by (String)`: Identifier of the user that created this Object

## Reading

The **Readings** collection contains all the data readings from a Sensor.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Object
* `sensor_id (ObjectID)`: Unique Identifier of the Sensor to which this Reading belongs.
* `timestamp (DateTime)`: Time where this reading took place
* `value (float)`: Value of the reading

## PipelineTemplate

The **PipelineTemplate** collection contains all the pipeline templates from which the
pipelines that later on will be used to run an experiments are generated.
The template includes all the default hyperparameter values, as well as the tunable
hyperparameter ranges.

### Fields

* `_id (ObjectID)`: Unique Identifier of this PipelineTemplate object
* `name (String)`: Name or code given to this Object
* `template (SubDocument)`: JSON representation of this pipeline template
* `insert_time (DateTime)`: Time when this Object was inserted
* `created_by (String)`: Identifier of the user that created this Object

## Pipeline

The **Pipeline** collection contains all the pipelines registered in the system, including
their details, such as the list of primitives and all the configured hyperparameter values.

### Fields

* `_id (ObjectID)`: Unique Identifier of this object
* `name (String)`: Name or code given to this Object
* `pipeline_template_id (ObjectID)`: Unique Identifier of the PipelineTemplate used to generate this pipeline
* `pipeline (SubDocument)`: JSON representation of this pipeline object
* `insert_time (DateTime)`: Time when this Object was inserted
* `created_by (String)`: Identifier of the user that created this Object

## MLTask

An **MLTask** is a specific Machine Learning Problem consisting on a prediction that
is to be made using a Pipeline.

### Fields

* `_id (ObjectID)`: Unique Identifier of this object
* `name (String)`: Name or code given to this Object
* `description (String)`: Short text description of this task
* `type (String)`: Type of Machine Learning Task
* `turbine_set (List[ObjectID])`: List of IDs of the Turbines to which this MLTask is applied
* `insert_time (DateTime)`: Time when this Object was inserted
* `created_by (String)`: Identifier of the user that created this Object

## Target

The **Target** collection contains the **MLTask** targets with their cutoff times.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Object
* `mltask_id (ObjectID)`: Unique Identifier of the MLTask to which this target belongs
* `turbine_id (ObjectID)`: Unique Identifier of the Turbine associated with this target
* `cutoff_time (DateTime)`: Time associated with this Target

## Experiment

An **Experiment** represents the process of trying and tuning a PipelineTemplate in order
to solve a MLTask.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Object
* `name (String)`: Name or code given to this Object
* `mltask_id (ObjectID)`: Unique Identifier of the MLTask to which this Experiment belongs
* `pipeline_template_id (ObjectID)`: Unique Identifier of the PipelineTemplate used in this Experiment
* `sensor_set (List[ObjectID])`: List of IDs of the Sensors used for this Experiment
* `cv_folds (integer)`: Number of folds used for Cross Validation
* `stratified (bool)`: Whether the Cross Validation was stratified or not
* `random_state (integer)`: Random State used for the Cross Validation shuffling
* `metric (string)`: Name of the metric used
* `insert_time (DateTime)`: Time when this Object was inserted
* `created_by (String)`: Identifier of the user that created this Object

## ExperimentRun

An **ExperimentRun** represents a single execution of an Experiment.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Object
* `experiment_id (ObjectID - Foreign Key)`: Unique Identifier of the Experiment
* `start_time (DateTime)`: When the execution started
* `end_time (DateTime)`: When the execution ended
* `software_versions (List of Strings)`: version of each python dependency installed in the
*virtualenv* when the execution took place
* `budget_type (String)`: Type of budget used (time or number of iterations)
* `budget_amount (Integer)`: Budget amount
* `status (String)`: Whether the ExperimentRun is still running, finished successfully or failed
* `insert_time (DateTime)`: Time when this Object was inserted
* `created_by (String)`: Identifier of the user that created this Object

## PipelineRun

A **PipelineRun** represents a single execution of a Pipeline instance over a MLTask.

It contains information about whether the execution was successful or not, when it started
and ended and the cross validation score obtained.

### Fields

* `_id (ObjectID)`: Unique Identifier of this Object
* `experimentrun_id (ObjectID)`: Unique Identifier of the ExperimentRun to which this PipelineRun belongs
* `pipeline_id (ObjectID)`: Unique Identifier of the Pipeline
* `start_time (DateTime)`: When the execution started
* `end_time (DateTime)`: When the execution ended
* `score (float)`: Cross Validation score obtained
* `status (String)`: Whether the Signalrun is still running, finished successfully or failed
* `insert_time (DateTime)`: Time when this Object was inserted

# CSV Format

As explained in a previous section, the input expected by the **GreenGuard** system consists of
two tables which need to be passed as `pandas.DataFrame` objects:

* The `target_times` table, which containing the specification of the problem that we are solving
  in the form of training examples with a `turbine_id`, a `cutoff_time` and a `target` value.
* The `readings` table, which contains the signal readings from the different sensors, with
  `turbine_id`, `signal_id`, `timestamp` and `value` fields.

However, in most scenarios the size of the available will far exceed the memory limitations
of the system on which **GreenGuard** is being run, so loading all the data in a single
`pandas.DataFrame` will not be possible.

In order to solve this situation, **GreenGuard** provides a [CSVLoader](
https://d3-ai.github.io/GreenGuard/api/greenguard.loaders.csv.html#greenguard.loaders.csv.CSVLoader)
class which can be used to load data from what we call the **Raw Data Format**.

## Raw Data Format

The **Raw Data Format** consists on a collection of CSV files stored in a single folder with the
following structure:

* All the data from all the turbines is inside a single folder, which here we will call `readings`.
* Inside the `readings` folder, one folder exists for each turbine, named exactly like the turbine:
    * `readings/T001`
    * `readings/T002`
    * ...
* Inside each turbine folder one CSV file exists for each month, named `%Y-%m.csv`.
    * `readings/T001/2010-01.csv`
    * `readings/T001/2010-02.csv`
    * `readings/T001/2010-03.csv`
    * ...
* Each CSV file contains three columns:
    * `signal_id`: name or id of the signal.
    * ``timestamp``: timestamp of the reading formatted as ``%m/%d/%y %H:%M:%S``.
    * `value`: value of the reading.

This is an example of what a CSV contents look like:

|    | signal_id   | timestamp         |   value |
|----|-------------|-------------------|---------|
|  0 | S1          | 01/01/01 00:00:00 |       1 |
|  1 | S1          | 01/01/01 12:00:00 |       2 |
|  2 | S1          | 01/02/01 00:00:00 |       3 |
|  3 | S1          | 01/02/01 12:00:00 |       4 |
|  4 | S1          | 01/03/01 00:00:00 |       5 |
|  5 | S1          | 01/03/01 12:00:00 |       6 |
|  6 | S2          | 01/01/01 00:00:00 |       7 |
|  7 | S2          | 01/01/01 12:00:00 |       8 |
|  8 | S2          | 01/02/01 00:00:00 |       9 |
|  9 | S2          | 01/02/01 12:00:00 |      10 |
| 10 | S2          | 01/03/01 00:00:00 |      11 |
| 11 | S2          | 01/03/01 12:00:00 |      12 |

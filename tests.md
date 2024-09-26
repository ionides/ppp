# Quantitative tests of pypomp

We are concerned with accuracy tests to make sure that the code gives correct answers (up to small Monte Carlo error) in situations where this is knowable.
We are also concerned with performance benchmark tests. This can involve measuring time, memory requirement, or iterations to convergence for a maximization algorithm.
Put together, we call these __quantitative tests__, or simply __quant tests__, to distinguish them from unit tests.
Our unit tests, in pypomp:pypomp/test, are quick tests designed to check that the code is not broken, and to ensure we are told if numerical results change.

The quant tests are in in a different repository, pypomp:quant.
There are arguments for and against putting them in the package reposity, pypomp:pypomp. 
Overall, it is better to keep the core code repository small and focused on the package itself.
The quant tests may become large.
The tests don't necessarily need to be run often.
If we test for correctness occasionally, it is enough to check more quickly and frequently that the results on small examples have not changed. 

## Structure of the tests

We propose many different short tests, each of which can be run independently.
Each test produces an html report.
An index links these results.
At a later date, some of these tests could be selected for a tutorial or a software announcement publication.

Each test has its own directory, within which we have a standard file structure.

* code.py. The test code, which can be run on greatlakes, or wherever

* code.sbat. A slurm batch file for running code.py on greatlakes

* results. A directory with saved results from code.py

* report.ipynb. A report presenting and discussing the results pulled in from the results directory

* report.html. The rendering of report.ipynb

* Makefile. To help automate the building of report.html

* README. A brief overview of the test

Each test should report:

1. The pypomp/Python/Jax versions used

2. The run time for each calculation

3. Other results

## Content of the tests





Pypomp is now tested on greatlakes GPUs, as described at https://ionides.github.io/ppp/.

We have good unit testing to make sure the code is not broken.

We should have 

  (i) The Linear-Gaussian model should give a log-likelihood consistent with the Kalman filter.

  (ii) The Dhaka model should give a log-likelihood consistent with pomp (which is itself checked against the Kalman filter).

These test pfilter and the two test models built into the package. Once that is done, we can also do:

  (iii) The maximization algorithms should find good approximations to the maxima for these examples.

We should also have performance tests. An ipynb document can write a report on how performance (run-time) scales with various factors. The report can be re-run after proposed edits. This is important because jax speed-up can be quite fragile, depending in subtle ways on coding decisions. So we need a framework for exploring performance.

Some draft documents that might later be included in pypomp/pypomp-docs or pypomp/pypomp.





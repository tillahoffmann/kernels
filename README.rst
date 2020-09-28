kernels
=======

.. image:: https://github.com/tillahoffmann/kernels/workflows/CI/badge.svg
  :target: https://github.com/tillahoffmann/kernels/actions

A package to (a) infer conditionally-independent edge models from ego networks collected in surveys and (b) evaluate a family of segregation measures for individuals, pairs of individuals, and society as a whole. See "`Inference of a universal social scale and segregation measures using social connectivity kernels <https://arxiv.org/abs/2008.05337>`__" for a detailed discussion.

Obtaining data for reproducing results
--------------------------------------

We cannot share data through this repository for licensing reasons, but the data can be obtained from their original sources. We provide download scripts for data that do not require the user to log on to a website, and provide instructions on how to obtain the data otherwise. Run ``make data`` from the root directory of this repository to download data and learn more.

* We use the Stata format of the General Social Survey (GSS). See `here <https://gss.norc.org/>`__ for more details.
* We use the Stata format of the British Household Panel Survey (BHPS) and Understanding Society (US) effort, and you will have to obtain data files from the UK Data Service. See `here <http://doi.org/10.5255/UKDA-SN-6614-13>`__ for more details.
* We use the Stata format of the American Life Panel (ALP). See `here <https://alpdata.rand.org/index.php?page=data&p=showsurvey&syid=86>`__ for more details. Please contact the authors to obtain a copy of the data which should be placed at ``data/ALP_MS86_2014_12_01_11_06_48_weighted.dta``.

Please run ``make data`` after having placed all archives in the correct location to verify that all required data are available.

Running the analysis
--------------------

After obtaining the necessary data, please run ``make figures`` to reproduce the figures presented in the publication. Intermediate files and figures that may aid debugging the inference will be created in the ``workspace`` folder (which is not tracked by source control).

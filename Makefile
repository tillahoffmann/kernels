.PHONY : data tests clean docs build figures

# Code stuff =======================================================================================

# Build requirements, tests, and documentation
build : flake8 tests docs

docs :
	sphinx-build -W docs docs/_build

# Compile python dependencies
requirements.txt : requirements.in setup.py
	pip-compile -v
	pip-sync

tests : requirements.txt
	pytest -v --cov=kernels --cov-report=term-missing --cov-report=html

flake8 : requirements.txt
	flake8

clean :
	rm -Rf htmlcov .pytest_cache workspace docs/_build

# Data =============================================================================================
data : data/gss data/bhps-us data/alp

# American Life Panel-------------------------------------------------------------------------------
data/alp : data/ALP_MS86_2014_12_01_11_06_48_weighted.dta

data/ALP_MS86_2014_12_01_11_06_48_weighted.dta :
	$(info Please contact the authors to obtain the data.)
	false

# General Social Survey ----------------------------------------------------------------------------
data/gss : data/GSS2004.dta data/GSS_Codebook.pdf

data/2004_stata.zip :
	mkdir -p $(dir $@)
	curl -o $@ https://gss.norc.org/documents/stata/2004_stata.zip

data/gss_codebook.zip :
	mkdir -p $(dir $@)
	curl -o $@ https://gss.norc.org/Documents/codebook/gss_codebook.zip

data/GSS2004.dta : data/2004_stata.zip
	unzip $< -d $(dir $@)
	touch $@

data/GSS_Codebook.pdf : data/gss_codebook.zip
	unzip $< -d $(dir $@)
	touch $@

# BHPS and Understanding Society -------------------------------------------------------------------
data/bhps-us : data/UKDA-6614-stata/6614_file_information.rtf

data/6614stata_18FA55F5E32F54B019743683E88E036D_V1.zip :
	$(info Please obtain the archive $@ from http://doi.org/10.5255/UKDA-SN-6614-13.)
	false

data/UKDA-6614-stata/6614_file_information.rtf : data/6614stata_18FA55F5E32F54B019743683E88E036D_V1.zip
	unzip $< -d data/
	# Need to touch the file to update timestamps to make `make` happy
	touch $@

# Figures ==========================================================================================

figures :

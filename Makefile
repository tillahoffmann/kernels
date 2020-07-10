# Set up different targets and datasets ============================================================

USOC_DATASETS = usoc_c usoc_f usoc_bb usoc_bd usoc_bf usoc_bh usoc_bj usoc_bl usoc_bn usoc_bp usoc_br
DATASETS = gss alp synthetic $(USOC_DATASETS)

USOC_TARGETS = $(addprefix workspace/,$(USOC_DATASETS))
INFERENCE_TARGETS = $(addprefix workspace/,$(DATASETS))

inference : $(INFERENCE_TARGETS)

.PHONY : data tests clean docs build figures $(INFERENCE_TARGETS)

SEED ?= 0
LOG_LEVEL ?= info
INFER_CMD = python scripts/infer.py --seed=$(SEED) --log-level=$(LOG_LEVEL)
NUM_DISTANCE_SAMPLES = 1000000

# General Social Survey/American Life Panel/Research Now -------------------------------------------

workspace/gss workspace/alp workspace/rn_uk1 : workspace/% :
	$(INFER_CMD) $*

# Understanding Society and British Household Panel Survey -----------------------------------------

workspace/usoc : $(USOC_TARGETS)

$(USOC_TARGETS) : workspace/% : workspace/uk_distance_samples-$(NUM_DISTANCE_SAMPLES).txt
	$(INFER_CMD) --distance-filename=$< $*

# Generate distance samples
workspace/uk_distance_samples-$(NUM_DISTANCE_SAMPLES).txt : \
		data/lsoa_boundaries_2011/LSOA_2011_EW_BGC_V2.shp \
		data/lsoa_boundaries_2011/SOA2011.shp \
		data/lsoa_boundaries_2011/SG_DataZone_Bdry_2011.shp \
		data/lsoa_population_2012/mid-2012-lsoa-syoa-unformatted-file.xls \
		data/lsoa_population_2012/2012-sape-t2a-corrected.xlsx \
		data/lsoa_population_2012/SAPE_SOA_0114.xls
	python scripts/sample_uk_distances.py --log-level=$(LOG_LEVEL) $(NUM_DISTANCE_SAMPLES)

# Simulations --------------------------------------------------------------------------------------
workspace/synthetic :
	python scripts/infer.py --log-level=$(LOG_LEVEL) --seed=0:250 --num-samples=2 synthetic

# Attribute-feature plots
PICKLE_FILES = $(filter-out workspace/synthetic/%,$(wildcard workspace/**/*.pkl))
AF_FILES = $(PICKLE_FILES:pkl=html)

debug :
	echo $(PICKLE_FILES)

af-plots : $(AF_FILES)

$(AF_FILES) : workspace/%.html : workspace/%.pkl scripts/attribute-feature-plots.ipynb
	filename=../$< jupyter nbconvert --execute --ExecutePreprocessor.timeout=-1 \
		--output-dir=. --output=$@ scripts/attribute-feature-plots.ipynb


# Code stuff =======================================================================================

# Build requirements, tests, and documentation
build : flake8 tests docs

docs :
	sphinx-build -W docs docs/_build

# Compile python dependencies
requirements.txt : requirements.in setup.py
	pip-compile -v
	pip-sync

tests :
	pytest -v --cov=kernels --cov-report=term-missing --cov-report=html

flake8 :
	flake8

clean_all :
	rm -Rf htmlcov .pytest_cache workspace docs/_build

clean_pkl :
	rm $(PICKLE_FILES)

clean_html :
	rm -f $(AF_FILES)

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

FIGURES = manuscript/embedding.pdf manuscript/credible-coverage.pdf \
	manuscript/coefficients-age-equivalent.pdf manuscript/segregation.pdf

figures : $(FIGURES)

$(FIGURES) : scripts/figures.ipynb workspace/gss workspace/usoc workspace/alp
	jupyter nbconvert --execute --ExecutePreprocessor.timeout=-1 \
		--output-dir=. --output=workspace/figures.html $<

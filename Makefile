# Shell variables which may customize behaviour of this Makefiles:
# * ADDITIONAL_PEX_OPTIONS
# 	additional flags which can be passed to pex tool to be used while building distribution files
# * OPTIONAL_FEATURES
# 	space seperated list of optional features to be included in the build of pex files;
# 	optional features list:
# 	* kafka_storage

PEX_OPTIONS = -v -R component-licenses --cache-dir=.pex-build $(ADDITIONAL_PEX_OPTIONS)
ENV_SAFE = env PYTHONPATH=. INCLUDE_UNSAFE_CONFLUENT_KAFKA_WHEEL=no
ENV_UNSAFE = env PYTHONPATH=. INCLUDE_UNSAFE_CONFLUENT_KAFKA_WHEEL=yes

OPTIONAL_MODULES =
ifeq ($(OPTIONAL_FEATURES),kafka_storage) 
	OPTIONAL_MODULES = 'confluent-kafka-python'
endif


# Do not really on artifacts created by make for all targets.
.PHONY: all venv flake8 bandit unit wca_package bandit_pex wrapper_package clean tests check dist

all: venv check dist

venv:
	@echo Preparing virtual enviornment using pipenv.
	pipenv --version
	env PIPENV_QUIET=true pipenv install --dev

flake8:
	@echo Checking code quality.
	pipenv run flake8 wca tests example workloads

bandit:
	@echo Checking code with bandit.
	pipenv run bandit -r wca -s B101 -f html -o wca-bandit.html

bandit_pex:
	@echo Checking pex with bandit.
	unzip dist/wca.pex -d dist/wca-pex-bandit
	pipenv run bandit -r dist/wca-pex-bandit/.deps -s B101 -f html -o wca-pex-bandit.html || true
	rm -rf dist/wca-pex-bandit

unit:
	@echo Running unit tests.
	env PIPENV_QUIET=true pipenv install flask
	pipenv run env PYTHONPATH=.:workloads/wrapper pytest --cov-report term-missing --cov=wca tests --ignore=tests/e2e/test_wca_metrics.py
	env PIPENV_QUIET=true pipenv uninstall flask

junit:
	@echo Running unit tests.	
	env PIPENV_QUIET=true pipenv install flask
	pipenv run env PYTHONPATH=.:workloads/wrapper pytest --cov-report term-missing --cov=wca tests --junitxml=unit_results.xml -vvv -s --ignore=tests/e2e/test_wca_metrics.py
	env PIPENV_QUIET=true pipenv uninstall flask

wca_package_in_docker: WCA_IMAGE := wca
wca_package_in_docker: WCA_TAG := $(shell git rev-parse HEAD)
wca_package_in_docker:
	@echo Building wca pex file inside docker and copying to ./dist/wca.pex
	# target: standalone
	sudo docker build --target standalone -f Dockerfile -t $(WCA_IMAGE):$(WCA_TAG) .
	# Extract pex to dist folder
	rm -rf .cidfile && sudo docker create --cidfile=.cidfile $(WCA_IMAGE):$(WCA_TAG)
	CID=$$(cat .cidfile); \
	mkdir -p dist; \
	sudo docker cp $$CID:/usr/bin/wca.pex dist/ && \
	sudo docker rm $$CID && \
	sudo chown -R $$USER:$$USER dist/wca.pex && sudo rm .cidfile
	@echo WCA image name is: $(WCA_IMAGE):$(WCA_TAG)
	@echo WCA pex file: dist/wca.pex

wca_package_in_docker_with_kafka: WCA_IMAGE := wca
wca_package_in_docker_with_kafka: WCA_TAG := $(shell git rev-parse HEAD)
wca_package_in_docker_with_kafka:
	@echo "Building wca pex (version with Kafka) file inside docker and copying to ./dist/wca.pex"
	# target: standalone
	sudo docker build -f Dockerfile.kafka -t $(WCA_IMAGE):$(WCA_TAG) .
	# Extract pex to dist folder
	rm -rf .cidfile && sudo docker create --cidfile=.cidfile $(WCA_IMAGE):$(WCA_TAG)
	CID=$$(cat .cidfile); \
	mkdir -p dist; \
	sudo docker cp $$CID:/wca/dist/wca.pex dist/ && \
	sudo docker rm $$CID && \
	sudo chown -R $$USER:$$USER dist/wca.pex && sudo rm .cidfile
	@echo WCA image name is: $(WCA_IMAGE):$(WCA_TAG)
	@echo WCA pex file: dist/wca.pex

wca_docker_devel: WCA_IMAGE ?= wca
wca_docker_devel: WCA_TAG ?= devel
wca_docker_devel: REPO ?= 100.64.176.12:80/
wca_docker_devel:
	@echo "Preparing development WCA container (only source code without pex)"
	sudo docker build --target devel -f Dockerfile -t $(REPO)$(WCA_IMAGE):$(WCA_TAG) .
	@echo WCA image name is: ${REPO}$(WCA_IMAGE):$(WCA_TAG)
	@echo Push: sudo docker push ${REPO}$(WCA_IMAGE):$(WCA_TAG)
	@echo Run: sudo docker run --privileged -ti --rm ${REPO}$(WCA_IMAGE):$(WCA_TAG) -0 -c /wca/configs/extra/static_measurements.yaml


wca_package:
	@echo Building wca pex file.
	-rm .pex-build/wca*
	-rm -rf .pex-build
	-rm dist/wca.pex
	-rm -rf wca.egg-info
	pipenv run $(ENV_SAFE) pex . $(OPTIONAL_MODULES) $(PEX_OPTIONS) -o dist/wca.pex -m wca.main:main
	./dist/wca.pex --version

wca_package_unsafe:
	@echo Building wca pex file.
	-rm .pex-build/wca*
	-rm -rf .pex-build
	-rm dist/wca.pex
	-rm -rf wca.egg-info
	pipenv run $(ENV_UNSAFE) pex . $(OPTIONAL_MODULES) $(PEX_OPTIONS) -o dist/wca.pex -m wca.main:main
	./dist/wca.pex --version

wrapper_package:
	@echo Building wrappers pex files.
	-sh -c 'rm -f .pex-build/*wrapper.pex'
	-rm -rf .pex-build
	pipenv run $(ENV_UNSAFE) pex . -D workloads/wrapper $(PEX_OPTIONS) -o dist/wrapper.pex -m wrapper.wrapper_main
	pipenv run $(ENV_UNSAFE) pex . -D workloads/wrapper $(PEX_OPTIONS) -o dist/example_workload_wrapper.pex -m wrapper.parser_example_workload
	pipenv run $(ENV_UNSAFE) pex . -D workloads/wrapper $(PEX_OPTIONS) -o dist/specjbb_wrapper.pex -m wrapper.parser_specjbb
	pipenv run $(ENV_UNSAFE) pex . -D workloads/wrapper $(PEX_OPTIONS) -o dist/ycsb_wrapper.pex -m wrapper.parser_ycsb
	pipenv run $(ENV_UNSAFE) pex . -D workloads/wrapper $(PEX_OPTIONS) -o dist/rpc_perf_wrapper.pex -m wrapper.parser_rpc_perf
	pipenv run $(ENV_UNSAFE) pex . -D workloads/wrapper $(PEX_OPTIONS) -o dist/tensorflow_benchmark_training_wrapper.pex -m wrapper.parser_tensorflow_benchmark_training
	pipenv run $(ENV_UNSAFE) pex . -D workloads/wrapper $(PEX_OPTIONS) -o dist/tensorflow_benchmark_prediction_wrapper.pex -m wrapper.parser_tensorflow_benchmark_prediction
	pipenv run $(ENV_UNSAFE) pex . -D workloads/wrapper $(PEX_OPTIONS) -o dist/mutilate_wrapper.pex -m wrapper.parser_mutilate
	pipenv run $(ENV_UNSAFE) pex . -D workloads/wrapper $(PEX_OPTIONS) -o dist/cassandra_stress_wrapper.pex -m wrapper.parser_cassandra_stress
	pipenv run $(ENV_UNSAFE) pex . -D workloads/wrapper $(PEX_OPTIONS) -o dist/stress_ng_wrapper.pex -m wrapper.parser_stress_ng
	pipenv run $(ENV_UNSAFE) pex . -D workloads/wrapper $(PEX_OPTIONS) -o dist/memtier_benchmark_wrapper.pex -m wrapper.parser_memtier
	./dist/wrapper.pex --help >/dev/null

check: flake8 bandit unit

dist: wca_package wrapper_package

clean:
	@echo Cleaning.
	rm -rf .pex-build
	rm -rf wca.egg-info
	rm -rf dist
	pipenv --rm

.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

# DOCKER TARGET
.PHONY: docker-login
docker-login:
	docker login

.PHONY: docker-build
docker-build:
	docker build -t greenguard-jupyter .

.PHONY: docker-push
docker-push: docker-login docker-build
	@$(eval VERSION := $(shell python -c 'import greenguard; print(greenguard.__version__)'))
	docker tag greenguard-jupyter signals-dev/greenguard:$(VERSION)
	docker push signals-dev/greenguard:$(VERSION)

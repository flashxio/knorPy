# Copyright 2016 neurodata (http://neurodata.io/)
# Written by Disa Mhembere (disa@jhu.edu)
#
# This file is part of knor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

HAS_PY3 := $(shell which python3)
ifeq ($(HAS_PY3),)
	FILES := py2
else
	FILES := py2 py3
endif

all: $(FILES)

py2:
	python setup.py build
py3:
	python3 setup.py build

install:
	#python setup.py install
	python3 setup.py install

dist:
	python setup.py sdist

up:
	twine upload dist/*

up-test:
	twine upload dist/* -r testpypi
	pip install -i https://testpypi.python.org/pypi knor

clean:
	rm -rf build
	rm -f MANIFEST
	rm -rf dist
	rm -f *.pyc
	rm -rf __pycache__
	rm -rf knor/__pycache__

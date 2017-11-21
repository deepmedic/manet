========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |docs| image:: https://readthedocs.org/projects/manet/badge/?style=flat
    :target: https://readthedocs.org/projects/manet
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/jonasteuwen/manet.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/jonasteuwen/manet

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/jonasteuwen/manet?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/jonasteuwen/manet

.. |requires| image:: https://requires.io/github/jonasteuwen/manet/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/jonasteuwen/manet/requirements/?branch=master

.. |codecov| image:: https://codecov.io/github/jonasteuwen/manet/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/jonasteuwen/manet

.. |version| image:: https://img.shields.io/pypi/v/manet.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/manet

.. |commits-since| image:: https://img.shields.io/github/commits-since/jonasteuwen/manet/v0.0.1.svg
    :alt: Commits since latest release
    :target: https://github.com/jonasteuwen/manet/compare/v0.0.1...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/manet.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/manet

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/manet.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/manet

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/manet.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/manet


.. end-badges

Manet is collection of tools to experiment with deep learning and medical images.

* Free software: BSD license

Installation
============

::

    pip install manet

Documentation
=============

https://manet.readthedocs.io/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox

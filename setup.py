#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

try:
    with open('README.md', encoding='utf-8') as readme_file:
        readme = readme_file.read()
except IOError:
    readme = ''

try:
    with open('HISTORY.md', encoding='utf-8') as history_file:
        history = history_file.read()
except IOError:
    history = ''

install_requires = [
    'baytune>=0.4.0,<0.5',
    'ml-stars>=0.1.0',
    'mlblocks>=0.4.0,<0.5',
    'pymongo>=3.7.2,<4',
    'scikit-learn>=0.21,<1.2',
    'tqdm<4.50.0,>=4.36.1',
    'scipy>=1.0.1,<2',
    'numpy>=1.16.0,<1.19',
    'pandas>=1,<2',
    'tensorflow>=2,<2.3',
    'partd>=1.1.0,<2',
    'fsspec>=0.8.5,<0.9',
    'dask>=2.6.0,<3',
    'tabulate>=0.8.3,<0.9',
    'xlsxwriter>=1.3.6,<1.4',
    # fix conflicts
    'protobuf<4',
    'importlib-metadata<5',
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

tests_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
    'jupyter>=1.0.0,<2',
    'rundoc>=0.4.3,<0.5',
]

development_requires = [
    # general
    'bumpversion>=0.5.3,<0.6',
    'pip>=9.0.1',
    'watchdog>=0.8.3,<0.11',

    # docs
    'm2r>=0.2.0,<0.3',
    'nbsphinx>=0.5.0,<0.7',
    'Sphinx>=1.7.1,<3',
    'sphinx_rtd_theme>=0.2.4,<0.5',
    'docutils>=0.14,<0.18',
    'autodocsumm>=0.1.10',
    'markupsafe<2.1.0',
    'Jinja2>=2,<3',

    # style check
    'flake8>=3.7.7,<4',
    'isort>=4.3.4,<5',

    # fix style issues
    'autoflake>=1.1,<2',
    'autopep8>=1.4.3,<2',
    'importlib-metadata<5',

    # distribute on PyPI
    'twine>=1.10.0,<4',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1,<6',
    'tox>=2.9.1,<4',
]

setup(
    author='MIT Data To AI Lab',
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description='AutoML for Time Series.',
    entry_points={
        'mlblocks': [
            'pipelines=draco:MLBLOCKS_PIPELINES',
            'primitives=draco:MLBLOCKS_PRIMITIVES'
        ],
    },
    extras_require={
        'test': tests_require,
        'dev': development_requires + tests_require,
    },
    include_package_data=True,
    install_requires=install_requires,
    keywords='wind machine learning draco',
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    name='draco-ml',
    packages=find_packages(include=['draco', 'draco.*']),
    python_requires='>=3.6,<3.9',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/sintel-dev/Draco',
    version='0.3.0',
    zip_safe=False,
)

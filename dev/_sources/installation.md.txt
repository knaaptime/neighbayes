# Installation

Currently, neighbayes supports Python >= [3.12]. Please make sure that you are operating in a Python 3 environment.

## Installing a released version

`neighbayes` is available on both conda and pip, and can be installed with any of

```bash
conda install -c conda-forge neighbayes
```

or

```bash
pixi add neighbayes
```

or

```bash
pip install neighbayes
```

## Installing a development from source

For working with a development version, we recommend [miniforge] or [pixi]. To get started, clone this repository or download it manually then `cd` into the directory and run the following commands:

**using conda**

```bash
conda env create -f environment.yml
conda activate neighbayes
pip install -e .
```

**using pixi**

*note*: as of this writing, pixi does not support relative paths (like "."), hence the expansion using the environment variable `$PWD`

```bash
pixi init --import environment.yml
pixi add --pypi --editable  "neighbayes @ file://$PWD"
```

You can also [fork] the [knaaptime/neighbayes] repo and create a local clone of your fork. By making changes to your local clone and submitting a pull request to [knaaptime/neighbayes], you can contribute to the neighbayes development.

## Building the documentation

```bash
conda env create -f docs/docs_env.yml
conda activate neighbayes
cd docs && make html
```

Every notebook under `docs/source/` is executed during the build
(`nb_execution_mode = "force"`), and an error in any of them fails the build.
Expect a full build to take tens of minutes.

[3.12]: https://docs.python.org/3.12/
[miniforge]: https://github.com/conda-forge/miniforge
[fork]: https://help.github.com/articles/fork-a-repo/
[knaaptime/neighbayes]: https://github.com/knaaptime/neighbayes
[python package index]: https://pypi.org/project/neighbayes/
[pixi]: https://pixi.prefix.dev/latest/

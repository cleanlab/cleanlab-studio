from setuptools import setup
from config import __version__
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cleanlab-cli",
    author="Cleanlab",
    author_email="caleb@cleanlab.ai",
    url="https://github.com/cleanlab/cleanlab-cli",
    long_description=long_description,
    description="Command line interface for all things Cleanlab Studio",
    version=__version__,
    py_modules=["main"],
    install_requires=[
        "Click",
        "colorama",
        "pandas",
        "pyexcel",
        "pyexcel-xls",
        "pyexcel-xlsx",
        "sqlalchemy",
        "requests",
        "tqdm",
        "ijson",
    ],
    entry_points="""
        [console_scripts]
        cleanlab=cleanlab_cli.main:cli
    """,
)

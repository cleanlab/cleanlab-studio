from setuptools import setup
from config import PACKAGE_VERSION

setup(
    name="cleanlab-cli",
    version=PACKAGE_VERSION,
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
    ],
    entry_points="""
        [console_scripts]
        cleanlab=cleanlab_cli.main:cli
    """,
)

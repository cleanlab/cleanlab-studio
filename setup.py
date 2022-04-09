from setuptools import setup

VERSION_NO = "0.1"
setup(
    name="cleanlab-cli",
    version=VERSION_NO,
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
    ],
    entry_points="""
        [console_scripts]
        cleanlab=cleanlab_cli.main:cli
    """,
)

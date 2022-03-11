from setuptools import setup

setup(
    name="cleanlab-cli",
    version="0.1",
    py_modules=["hello"],
    install_requires=[
        "Click",
        "colorama",
        "pandas",
        "pyexcel",
        "pyexcel-xls",
        "pyexcel-xlsx",
        "sqlalchemy",
    ],
    entry_points="""
        [console_scripts]
        cleanlab=dataset:cli
    """,
)

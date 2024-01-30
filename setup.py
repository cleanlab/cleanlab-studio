from os import path

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Get version number and store it in __version__
exec(open("cleanlab_studio/version.py").read())

setup(
    name="cleanlab-studio",
    version=__version__,
    license="MIT",
    author="Cleanlab Inc",
    author_email="team@cleanlab.ai",
    description="Client interface for all things Cleanlab Studio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cleanlab/cleanlab-studio",
    project_urls={
        "Bug Tracker": "https://github.com/cleanlab/cleanlab-studio/issues",
        "Documentation": "https://help.cleanlab.ai",
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    keywords="cleanlab",
    packages=find_packages(exclude=[]),
    py_modules=["main"],
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.8.1",
        "Click>=8.1.0,<=8.1.3",
        "colorama>=0.4.4",
        "nest_asyncio>=1.5.0",
        "pandas==2.*",
        "pyexcel>=0.7.0",
        "pyexcel-xls>=0.7.0",
        "pyexcel-xlsx>=0.6.0",
        "requests>=2.27.1",
        "tqdm>=4.64.0",
        "ijson>=3.1.4",
        "jsonstreams>=0.6.0",
        "semver>=2.13.0,<3.0.0",
        "Pillow>=9.2.0",
        "typing_extensions>=4.2.0",
        "openpyxl>=3.0.0,!=3.1.0",
        "validators>=0.20.0",
    ],
    entry_points="""
        [console_scripts]
        cleanlab=cleanlab_studio.cli:main
    """,
)

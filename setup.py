from setuptools import setup

setup(
    name='cleanlab-cli',
    version='0.1',
    py_modules=['hello'],
    install_requires=[
        'Click',
    ],
    entry_points="""
        [console_scripts]
        cl=cleanlab_pro:main
    """
)
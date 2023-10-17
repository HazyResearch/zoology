from setuptools import setup

_REQUIRED = [

]

setup(
    name="zoology",
    version="0.0.1",
    description="",
    author="simran sabri",
    packages=["zoology"],
    install_requires=_REQUIRED,
    entry_points={
        'console_scripts': ['zg=zoology.cli:cli'],
    },
)
from distutils.core import setup
from setuptools import find_packages

setup(
    name="plot_calculator",
    version="0.0.1",
    author="eayin2",
    author_email="eayin2@gmail.com",
    packages=find_packages(),
    package_data={'': ['*.txt']},
    url="https://github.com/eayin2/plot_calculator",
    description="Python interactive shell calculator",
    install_requires=[
        "numpy",
        "matplotlib",
        "pyside2",
        "sympy",
        "scipy",
        "appdirs"],
    include_package_data=True,
    entry_points={
        "console_scripts": [
           "plot_calculator = plot_calculator.calculator:start_calculator",
    ]}
)

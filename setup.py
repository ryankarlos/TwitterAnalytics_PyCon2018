from setuptools import find_packages, setup

with open("README.md") as readme_file:
    long_description = readme_file.read()

setup(
    name="Twitter Analytics",
    version="0.1",
    description="Twitter Analytics and ML",
    long_description=readme(),
    url="https://github.com/ryankarlos/TwitterAnalytics_PyCon2018",
    author="Ryan Nazareth",
    author_email="ryankarlos@gmail.com",
    license="MIT",
    packages=find_packages(exclude=["tests", "notebooks", "examples"]),
)

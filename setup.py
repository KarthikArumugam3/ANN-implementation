from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    author="KarthikArumugam3",
    description="A small package for ANN classifier Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KarthikArumugam3/ANN-implementation",
    author_email="karthik131100@gmail.com",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        "tensorflow",
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas",
        "PyYAML"
    ]
)
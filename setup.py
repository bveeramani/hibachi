import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hibachi",
    version="0.0.1",
    author="Balaji Veeramani",
    author_email="bveeramani@berkeley.edu",
    description="Feature selection methods for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bveeramani/hibachi",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)

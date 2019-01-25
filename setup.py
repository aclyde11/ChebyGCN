import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ChebyGCN-aclyde",
    version="0.0.2",
    author="Austin Clyde",
    author_email="aclyde@uchicago.edu",
    description="Implements graph convolution keras layers based on Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst, Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering, Neural Information Processing Systems (NIPS), 2016.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aclyde11/ChebyGCN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2",
        "Operating System :: OS Independent",
    ],
)
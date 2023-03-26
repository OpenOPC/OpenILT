import setuptools

# setup(
#     name='adaptive_boxes',
#     version='0.1',
#     scripts=['adabox']
# )

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="adaptive-boxes",
    version="0.0.4",
    author="Juan Francisco Chango",
    author_email="jnfran92@gmail.com",
    description="Python package for rectangular decomposition of 2D scenes/binary images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jnfran92/adaptive-boxes",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        'Intended Audience :: Science/Research',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
            # "certifi==2019.3.9",
            # "cycler==0.10.0",
            # "kiwisolver==1.1.0",
            # "matplotlib==3.1.1",
            # "numpy==1.22.0",
            # "pandas==0.25.1",
            # "plyfile==0.7",
            # "pyparsing==2.4.2",
            # "python-dateutil==2.8.0",
            # "pytz==2019.2",
            # "scipy==1.3.1",
            # "six==1.12.0",
        ],
)

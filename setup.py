import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dl4nlp",
    version="0.0.1",
    author="Robert Jan Schlimbach, Mathieu Barthels, Guido Visser, Paul ten Kaate",
    description="repo for dl4nlp",
    long_description=long_description,
    url="https://github.com/robogast/DL4NLP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

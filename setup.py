"""Setups up the tgdp module."""

from setuptools import find_packages, setup


def get_description():
    """Get the description from the readme."""
    with open("README.md") as fh:
        long_description = ""
        header_count = 0
        for line in fh:
            if line.startswith("##"):
                header_count += 1
            if header_count < 2:
                long_description += line
            else:
                break
    return header_count, long_description


def get_version():
    """Get the tgdp version."""
    path = "tgdp/__init__.py"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


version = get_version()
header_count, long_description = get_description()

setup(
    name="tgdp",
    version=version,
    author="Anonymous",
    author_email="anon@ymous.com",
    description="Implementation of Temperature Guided Diffusion Planning",
    url="https://github.com/iclr26-submission24685/tgdp_submission",
    license="MIT",
    license_files=("LICENSE",),
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["Reinforcement Learning", "Diffusion", "Decision Making"],
    packages=find_packages(),
    include_package_data=True,
    package_data={},
    install_requires=[
        "setuptools",
    ],
)

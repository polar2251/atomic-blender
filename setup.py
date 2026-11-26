from setuptools import find_packages, setup

setup(
    name="atomic-blender",
    version="0.1.0",
    packages=find_packages(include=["atomic_blender*"]),
    install_requires=["ase>=3.22", "numpy>=1.22"],
    entry_points={
        "console_scripts": [
            "atomic-blender=atomic_blender.cli:main",
        ]
    },
)

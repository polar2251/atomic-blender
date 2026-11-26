from setuptools import find_packages, setup

setup(
    name="atomic-blender",
    version="0.1.0",
    license="MIT",
    packages=find_packages(include=["atomic_blender*"]),
    install_requires=["ase>=3.22", "numpy>=1.22", "pillow>=9.0"],
    package_data={
        "atomic_blender": [
            "styles/*.json",
            "addon/atomic_blender_studio/*.py",
        ],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "atomic-blender=atomic_blender.cli:main",
            "atomic-blender-gui=atomic_blender.gui:main",
        ]
    },
)

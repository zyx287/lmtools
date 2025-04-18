from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="lmtools",
        version="0.1.0",
        packages=find_packages(),  # No 'where' parameter needed
        install_requires=[
            "numpy",
            "zarr==2.18.2",
            "nd2",
            "napari",
            "opencv-python",
            "pillow",
            "scipy",
            "pandas",
            "pyyaml",
        ],
        extras_require={
            "cellpose": ["cellpose", "torch"],
            "dev": ["flake8", "pytest", "mypy"],
        },
        entry_points={
            'console_scripts': [
                'lmtools=lmtools.__main__:main',
            ],
        },
    )
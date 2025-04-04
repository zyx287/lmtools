from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="lmtools",
        version="0.0.1",
        packages=find_packages(),
        install_requires=[
            "numpy",
            "zarr==2.18.2",
            "nd2",
            "napari",
            "opencv-python",
            "pillow",
        ],
        entry_points={
            'console_scripts': [
                'lmtools=lmtools.__main__:main',
            ],
        },
    )
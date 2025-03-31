from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="lmtools",  # Name of your package
        version="0.0.1",  # Version of your package
        packages=find_packages(where="."),  # Automatically find all sub-packages
        package_dir={"": "."},
        # install_requires=[],
        entry_points={
            'console_scripts': [
                'lmtools-cli=lmtools.cli.__main__:main',
            ],
        },
    )

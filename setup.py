from setuptools import setup, find_packages

setup(
    name="pig-monitoring",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "PyYAML",
        "moviepy",
        "opencv-python<4.11",
        "psutil",
        "pandas",
        "numpy",
        "torch",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ],
)

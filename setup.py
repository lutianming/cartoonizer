from setuptools import setup

# Metadata goes in setup.cfg. These are here for GitHub's dependency graph.
setup(
    name="cartooner",
    install_requires=[
        "numpy",
        "scipy",
        "opencv-python",
    ]
)

from setuptools import find_packages, setup

setup(
    name="vit",
    version="0.1.0",
    description="VIT for Leaf Disease detection",
    author="divital-coder",
    author_email="divital2004@gmail.com", 
    url="https://github.com/divital-coder",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "pillow",
        "pyyaml",
        "tensorboard",
        "matplotlib",
        "scikit-learn",
        "pandas",
        "tqdm",
        "wandb",
        "timm",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)

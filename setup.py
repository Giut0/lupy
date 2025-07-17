from setuptools import setup, find_packages

setup(
    name="lupy",
    version="0.1.1",
    description="Camera Trap Video Classification Tool",
    author="Vito Proscia",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.0.0",
        "joblib>=1.2.0",
        "timm>=0.6.13",
        "opencv-python>=4.7.0",
        "typer>=0.9.0",
        "megadetector>=5.0.0",
        "pytesseract>=0.3.8"
    ],
    entry_points={
        "console_scripts": [
            "lupy = lupy.main:lupy",
        ],
    },
    package_data={
        "lupy": ["models/*", 'models/tessdata/*.traineddata'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

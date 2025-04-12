from setuptools import setup, find_packages

setup(
    name='openqstack',
    version='0.1.0',
    description='Modular, educational quantum error correction toolkit',
    author='Jaebum Eric Kim',
    author_email='erickim1492@gmail.com',
    packages=find_packages(),  # finds openqstack/
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Education',
    ],
)

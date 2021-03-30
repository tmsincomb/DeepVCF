from setuptools import setup, find_packages
from setuptools.extension import Extension
# from Cython.Build import cythonize
import numpy as np


with open('README.md') as infile:
    long_description = infile.read()


extensions = [
    Extension(
        "DeepVCF.cython_numpy.cython_np_array",
        ["DeepVCF/cython_numpy/cython_np_array.pyx"],
        include_dirs=[np.get_include()], # needed for cython numpy to_array
        # libraries=['',],
        # library_dirs=['', ], 
    ),
]


setup(
    name='deepvcf',
    version='1.0.0',  # major.minor.maintenance
    description='Variant Caller for Prokaryotic Genomes using TensorFlow',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tmsincomb/DeepVCF',
    author='Troy Sincomb',
    author_email='troysincomb@gmail.com',
    license='MIT',
    keywords='deep vcf deepvcf',
    packages=find_packages('DeepVCF'),
    # include_package_data=True,  # try this out: might be the reason packages didnt break since it wont run without this.
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 1 - ALPHA',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    #  TODO: add classifiers for machine learning/variant caller https://pypi.org/classifiers/
    install_requires=[
        'tf-nightly',  # tensoflow production usually lagging in python version compatablity
        'biopython',
        'pandas',
        'pysam',
        'scipy',
        'cython' ,
    ],
    entry_points={
        'console_scripts': [
            'deepvcf=DeepVCF.core:main',
        ],
    },
    setup_requires=["cython"],
    # ext_modules=cythonize(extensions),  # used for initial build. 
    ext_modules=extensions,  
)
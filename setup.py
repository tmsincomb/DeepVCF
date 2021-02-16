from setuptools import setup, find_packages

setup(
    name='deepvcf',
    version='0.0.1',
    description='Variant Caller for Prokaryotic Genomes using TensorFlow',
    long_description='',
    url='https://github.com/tmsincomb/DeepVCF',
    author='Troy Sincomb',
    author_email='troysincomb@gmail.com',
    license='MIT',
    keywords='deep vcf deepvcf',
    packages=find_packages('deepvcf'),
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 1 - ALPHA',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=[
        'tensorflow',
        'biopython',
        'pandas',
        'numpy',
        'pysam',
    ],
    entry_points={
        'console_scripts': [
            'deepvcf=DeepVCF.core:main',
        ],
    },
)
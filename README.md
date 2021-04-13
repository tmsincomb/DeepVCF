# What is DeepVCF?
A deep learning SNP variant caller aimed for Prokaryotic genomes!

# Prerequisites
### WARNING :: Tensorflow stable does not work with python 3.9 yet!
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)

# Install
```bash
git clone git@github.com:tmsincomb/DeepVCF.git
pip install -e ./DeepVCF
```

# Optional Install
### Creating *in silico* datasets
```bash
conda install -y dwgsim samtools bcftools bwa
```

# Usage: Init
```Python
from DeepVCF.core import DeepVCF

deepvcf = DeepVCF()
```

# Simple Usage: Variant Calling 
```python
deepvcf.train(reference_file, alignment_file, vcf_file)  # vcf treated as truth 
vcf_df = deepvcf.create_vcf(
    reference_file=query_ref_file,   
    alignment_file=query_align_file,
    output_folder='./',  
    output_prefix='my-variants'  # auto adds .deepvcf.vcf to end of file created
)
vcf_df.head() # shows pandas DataFrame for variant outputs
```

# Tutorials
[Recreating Example Datasets](./jupyter_nb/creating-data-for-usage-demo.ipynb)</br>
[Usage Demo with In Silico datasets](./jupyter_nb/tutorial.ipynb)</br>
[Model Validation with human datasets from GIAB](./jupyter_nb/GIAB-usage-demo.ipynb)</br>

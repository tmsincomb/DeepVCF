"""
Where all the other files converge.
This file should not be long and should be incredabliy human readable.
People should look at this and think; wow that's simpiler than I thought. We need this in research.
"""
from typing import Callable, Union

import numpy as np
from pysam import AlignmentFile

from .alignment import alignment
from .biopandas import pandas as pd
from .tools import pathing


class DeepVCF:

    class OptionError(Exception):
        pass

    def __init__(self,
                 reference_file: Union[list, str],
                 reads_file: Union[list, str],
                 alignment_file: Union[list, str], 
                 verbose: bool = True,):
        self.multi_tensor = self._preprocessing(reference_seq, target_seq, alignment_file)
        self. self._predict(self.multi_tensor)


    @staticmethod
    def __methods(_class: Callable) -> dict:
        methods = {
            func: getattr(_class, func)
            for func in dir(_class)
            if callable(getattr(_class, func)) and not func.startswith("__")  # ignore dunder methods
        }
        return methods

    def __call_method(self, _class: object, option: str) -> Callable:
        methods: dict = self.__methods(alignment)
        method = methods.get(option)
        if method is None:
            raise OptionError(f'Please choose from the following options for {str(_class)}: {list(methods)}')
        return method

    def _alignment(self, option: str = 'blastn', **kwargs) -> AlignmentFile:
        method = self.__call_method(alignment, option)
        return method(**kwargs)

    def _preprocessing(self, ref_path, format):
        ref_seq_records = list(SeqIO.parse(pathing(ref_path), format=format))
        if len(ref_seq_records) != 1:
            raise ValueError('Reference needs to be a single sequence')
        else:
            ref_seq = ref_seq_records[0]

    def _model(self):
        method = self.__call_method(alignment, option)
        return method(**kwargs)

    def _train(self):
        pass

    def _postprocessing(self):
        pass

    def validation_graphs(self):
        pass

    def validation_data(self):
        pass


def main():
    pass

if __name__ == '__main__':
    main()



### Dream usage
# import DeepVCF


# deepvcf = DeepVCF()

# # Create VCF
# df_vcf = deepvcf.create_vcf('ref.fastq', 'reads.fastq', 'alignment.bam')
# # Notable instances
# deepvcf.header  # Text of real VCF header
# deepvcf.vcf  # VCF DatatFrame
# deepvcf.annotated_vcf  # VCF DatatFrame with extra columns for variant accuracies 
# deepvcf.metrics  # Current models metrics

# # Saving
# deepvcf.save_vcf('destination.vcf')
# deepvcf.save_annotated_vcf('destination.annotated.vcf')

# # Alter Model
# deepvcf.alter_tensor_dimensions(width=33, height=100)
# deepvcf.alter_model(layers=['conv2d', 'maxpool', 
#                             'conv2d', 'maxpool', 
#                             'conv2d', 'maxpool', 
#                             'fc', 'fc'])

# # Custom layer
# conv2d = keras.layers.Conv2D(filters, kernel_size, **kwargs)
# deepvcf.alter_model(layers=[ conv2d,  'maxpool', 
#                             'conv2d', 'maxpool', 
#                             'conv2d', 'maxpool', 
#                             'fc', 'fc'])

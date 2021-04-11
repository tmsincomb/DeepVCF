"""
Where all the other files converge.
This file should not be long and should be incredabliy human readable.
People should look at this and think; wow that's simpiler than I thought. We need this in research.
"""
VERSION = '0.1.0'

from typing import Callable, Union

import numpy as np
from pysam import AlignmentFile

from .alignment import alignment
from .biopandas import pandas as pd
from .tools import pathing
from .preprocess import Preprocess
from .model import models
from .create_vcf import VCF
from .postprocess import Metrics


class DeepVCF:

    class OptionError(Exception):
        pass

    def __init__(self):
        self.pileup = None
        self.model = None
        self.vcf = None

    @staticmethod
    def __methods(_class: Callable) -> dict:
        methods = {
            func: getattr(_class, func)
            for func in dir(_class)
            if callable(getattr(_class, func)) and not func.startswith("__")  # ignore dunder methods
        }
        return methods

    def __call_method(self, _class: object, option: str) -> Callable:
        methods: dict = self.__methods(_class)
        method = methods.get(option)
        if method is None:
            raise self.OptionError(f'Please choose from the following options for {str(_class)}: {list(methods)}')
        return method

    def _alignment(self, option: str = 'blastn', **kwargs) -> AlignmentFile:
        method = self.__call_method(alignment, option)
        return method(**kwargs)

    def _preprocessing(self, reference_file, alignment_file, **kwargs):
        self.preprocess = Preprocess(reference_file, alignment_file, **kwargs)

    def _model(self, option: str = 'default_model', **kwargs):
        method = self.__call_method(models, option)
        self.model = method(**kwargs)

    def _postprocessing(self):
        pass

    def train(self, reference_file, alignment_file, window_size:int=15, **kwargs) -> None:
        self._preprocessing(reference_file, alignment_file, **kwargs)
        x_train, y_train, pos_array = self.preprocess.get_training_array(window_size)
        self._model(input_shape=x_train.shape[1:])
        self.model = models.train_model(self.model, x_train, y_train, **kwargs)

    def validation(self, real_vcf, predicted_vcf, **kwargs) -> None:
        self.metrics = Metrics(real_vcf, predicted_vcf, **kwargs).metrics
        return self.metrics

    def create_vcf(self, reference_file, alignment_file, output_folder, output_prefix, window_size:int=15, **kwargs) -> None:
        self._preprocessing(reference_file, alignment_file, **kwargs)
        x_test, y_test, pos_array = self.preprocess.get_training_array(window_size)
        alt_bases, genotypes = self.model.predict(x_test)
        ref_bases = self.preprocess.ref_seq
        ref = pd.read_seq(reference_file, format='fasta')
        reference_id = ref.id.tolist()[0]
        reference_name = ref.name.tolist()[0]
        vcf = VCF(output_folder, output_prefix, self.preprocess.pileup.shape[0], reference_name, **kwargs)
        return vcf.create_vcf(reference_name, reference_id, ref_bases, alt_bases, genotypes, pos_array)


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

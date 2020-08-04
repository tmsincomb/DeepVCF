"""
Align undefined amount of genomes to a single reference. That reference being what is closest to the target sequence.

This is to only act as an example for alignment. The goal was not to provide a perfect auto-alignment, but
rather to provide an easy one function does all for new people in the bioinformatics field.

Please expand on these functions or simply input a list of aligned bam/sam file paths to skip this file altogther.
"""
import multiprocessing
import os
from subprocess import check_output
import tempfile

from Bio.Blast.Applications import NcbiblastnCommandline
from pysam import AlignmentFile

from .biopandas import pandas as pd
from .tools import pathing

# https://docs.python.org/3/library/subprocess.html
# http://bowtie-bio.sourceforge.net/bowtie2/manual.shtml#end-to-end-alignment-example


class Alignment:

    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()

    def __repr__(self):
        return 'Alignment'

    @staticmethod
    def blastn(query: str, subject: str, verbose: bool = True) -> AlignmentFile:
        blastncmd = NcbiblastnCommandline(
            task='megablast',  # for the same specie
            num_threads=multiprocessing.cpu_count(),
            query=query,
            subject=subject,
            outfmt="17 SQ",  # sam format with sequences
        )
        if verbose:
            print(blastncmd)  # cmd to be ran
        stdout, stderr = blastncmd()  # init cmd
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, 'w') as tmp:  # create randomized tmp file
            tmp.write(stdout)  # populate tmp file
            # Silly pysam cant handle strings & sys.stdout will crash jupyter or spyder
            pysam_object = AlignmentFile(path)
        os.remove(path)  # remove tmp file
        return pysam_object


alignment = Alignment()

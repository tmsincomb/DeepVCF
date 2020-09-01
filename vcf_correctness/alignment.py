"""
Align undefined amount of genomes to a single reference. That reference being what is closest to the target sequence.

This is to only act as an example for alignment. The goal was not to provide a perfect auto-alignment, but
rather to provide an easy one function does all for new people in the bioinformatics field.

Please expand on these functions or simply input a list of aligned bam/sam file paths to skip this file altogther.
"""
import multiprocessing
import os
from subprocess import check_output, STDOUT
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
        fd, path = tempfile.mkstemp()
        blastncmd = NcbiblastnCommandline(
            task='megablast',  # for the same specie
            # num_threads=multiprocessing.cpu_count(),  # ignored when subj given :/
            query=query,
            subject=subject,
            out=path,
            outfmt="17 SQ",  # sam format with sequences
        )
        if verbose:
            print(blastncmd)  # cmd to be ran
        stdout, stderr = blastncmd()  # init cmd
        pysam_object = AlignmentFile(path)
        os.remove(path)  # remove tmp file
        return pysam_object
        # if verbose:
        #     print(blastncmd)  # cmd to be ran
        # stdout, stderr = blastncmd()  # init cmd
        # fd, path = tempfile.mkstemp()
        # with os.fdopen(fd, 'w') as tmp:  # create randomized tmp file
        #     tmp.write(stdout)  # populate tmp file
        #     # Silly pysam cant handle strings & sys.stdout will crash jupyter or spyder
        #     pysam_object = AlignmentFile(path)
        # os.remove(path)  # remove tmp file
        # return pysam_object

    def nucmer(self, query: str, subject: str, verbose: bool = True) -> AlignmentFile:
        fd, path = tempfile.mkstemp()
        cmd = ' '.join(list(map(str, [
            'nucmer',
            f'--sam-short {pathing(path)}',
            f'--threads {self.cpu_count}',
            pathing(subject),
            pathing(query),
        ])))
        if verbose:
            print(cmd)
        _ = check_output(cmd, shell=True, stderr=STDOUT)
        with open(path, 'r') as tmp:
            lines = tmp.readlines()
            start = 0
            for i, line in enumerate(lines):
                if line.startswith('@'):
                    start = i + 1
                else:
                    break
            content = ''.join(lines[start:])
            new_header = '@SQ\tSN:FakeHeader\tLN:0\n'
        with open(path, 'w') as tmp:
            tmp.write(new_header + content)
        pysam_object = AlignmentFile(path, 'r', check_header=False)
        print(path)
        # os.remove(path)  # remove tmp file
        return pysam_object


alignment = Alignment()

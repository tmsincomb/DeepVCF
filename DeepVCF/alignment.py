"""
Align undefined amount of genomes to a single reference. That reference being what is closest to the target sequence.

This is to only act as an example for alignment. The goal was not to provide a perfect auto-alignment, but
rather to provide an easy one function does all for new people in the bioinformatics field.

Please expand on these functions or simply input a list of aligned bam/sam file paths to skip this file altogther.
"""
import multiprocessing
import os
from subprocess import check_output, STDOUT, CalledProcessError
import tempfile
from typing import Union, List, Dict, Optional, Tuple
import sys 

from Bio.Blast.Applications import NcbiblastnCommandline
import pysam
from pysam import AlignmentFile

from .biopandas import pandas as pd
from .tools import pathing

# https://docs.python.org/3/library/subprocess.html
# http://bowtie-bio.sourceforge.net/bowtie2/manual.shtml#end-to-end-alignment-example


class Alignment:

    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.output = pathing('/tmp/')
    
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

    def bwa_mem(self, 
                reference: str,
                reads: Union[list, str],
                output_folder,
                output_prefix,
                verbose: bool = True, 
                return_pysam = True,
                **options) -> pysam.AlignmentFile:
        """bwa mem aligner with samtools sort/cleanup

        Args:
            reference (str): reference sequence of
            reads (Union[list, str]): query sequences; forward and reverse are tipical with illumina 
            output_folder ([type]): folder to dump sam, inter-steps, and cleaned/sorder bam file
            output_prefix ([type]): prefix to the sam+ files
            verbose (bool, optional): prints command ans stats. Defaults to True.

        Returns:
            pysam.AlignmentFile: object that can be parsed by DeepVCF.biopandas.read_seq
        """
        reference = pathing(reference)
        reads = ' '.join([str(pathing(read)) for read in reads]) if isinstance(reads, list) else pathing(reads)
        output = (pathing(output_folder) / output_prefix).with_suffix('.sam')
        hard_coded_options = {'-t': self.cpu_count}
        # option types to string with priority going to incoming options
        options = ' '.join([f'{k} {v}' for k, v in {**hard_coded_options, **options}.items()])
        # create an end result output_prefix.bam file that is cleaned and sorted.
        cmds = [
            # index reference fasta
            f'bwa index {reference}',
            # # align reads to reference 
            f'bwa mem {options} {reference} {reads} > {output}',
            # # compress sam file to a bam (binary) file
            f'samtools view -Shb {output} > {output.with_suffix(".unsorted.bam")}',
            # sort the alignments based on names of reads; pre-req for fixmate
            f'samtools sort -n {output.with_suffix(".unsorted.bam")} -o {output.with_suffix(".sortedname.bam")}',
            # fill in mate coordinates and insert size fields
            f'samtools fixmate -m {output.with_suffix(".sortedname.bam")} {output.with_suffix(".fixmate.sortedname.bam")}',
            # sort the alignments on coordinates
            f'samtools sort {output.with_suffix(".fixmate.sortedname.bam")} -o {output.with_suffix(".fixmate.sorted.bam")}',
            # remove duplicates 
            f'samtools markdup -r -s --threads {self.cpu_count} {output.with_suffix(".fixmate.sorted.bam")} {output.with_suffix(".bam")}',
            f'samtools view -b -F 4 {output.with_suffix(".bam")} > {output.with_suffix(".mapped.bam")}',
            f'samtools index -@ {self.cpu_count} {output.with_suffix(".mapped.bam")}',
            # remove large tmp files; rm cmd breaks check_ouput if inbetween cmds... TODO: use python tmp to avoid this.
            f'rm {output}',
            f'rm {output.with_suffix(".unsorted.bam")}',
            f'rm {output.with_suffix(".sortedname.bam")}',
            f'rm {output.with_suffix(".fixmate.sortedname.bam")}',
            f'rm {output.with_suffix(".fixmate.sorted.bam")}',
            f'rm {output.with_suffix(".bam")}',
        ]
        for cmd in cmds:
            if verbose: print(cmd)
            try:
                _ = check_output(cmd, shell=True)
            except CalledProcessError as e:
                sys.exit(e.output)
        # todo print -> samtools flagstat $DATA/SRR1544640_alignment_sorted.bam for verbose
        if return_pysam:
            pysam_object = AlignmentFile(f'{output.with_suffix(".mapped.bam")}', 'r', check_header=False)
            return pysam_object

alignment = Alignment()

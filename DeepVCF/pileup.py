import multiprocessing
import re 
import sys 

from pysam import AlignmentFile
from scipy.sparse import hstack, vstack, csr_matrix, csc_matrix, coo_matrix, save_npz, load_npz
import numpy as np 
from itertools import zip_longest
from Bio import SeqIO  # __init__ kicks in
try:
    from .tools import pathing
    from .cython_numpy.cython_np_array import to_array
except:
    # For debugging speed issues with pyinstrument
    # COMMAND: pyinstrument --show-all --outfile pileup.log pileup.py
    from DeepVCF import pathing, to_array


def get_ref_seq(reference_file: str, format: str = 'fasta'):
    reference_file = pathing(reference_file)
    seqrecords = list(SeqIO.parse(reference_file, format=format))
    if len(seqrecords) > 1:
        exit('ERROR :: Reference fasta must be a single complete sequence')
    return seqrecords[0].seq


def get_alignments(alignment_file, threads=multiprocessing.cpu_count()):
    return AlignmentFile(pathing(alignment_file), threads=threads)


class Pileup:
    """ """
    baseindex={
        'A':0, 'a':0,
        'C':1, 'c':1,
        'G':2, 'g':2,
        'T':3, 't':3,
        '*': None,
        'N': None, 'n': None,
        '': None,
    }
    def __repr__():
        return 'Pileup 2D matrix - REF/deletion + ALT + ALT/insertion'

    def __init__(self, 
                 reference_file, 
                 alignment_file, 
                 save_pileup_to_destination: str = None,
                 use_saved_pileup: str = None,
                 minimum_base_quality = 10,
                 minimum_mapping_quality = 50,
                 minimum_alignment_coverage: int = 200, 
                 minimum_variant_coverage: int = 10,
                 heterozygous_threshold: float = .15,
                 minimum_variant_radius: int = 12,
                 **kwargs,) -> None:
        # helper variables
        self.contig_start= None 
        self.contig_end = None
        # variant parameters
        self.minimum_variant_coverage = minimum_variant_coverage
        self.heterozygous_threshold = heterozygous_threshold  # minimum difference a secondary alt needs to be for a variant to be called. 
        self.minimum_variant_radius = minimum_variant_radius  # distance variants are allowed to be without being classified as complex
        # aligment parameters
        self.minimum_mapping_quality = minimum_mapping_quality
        self.minimum_base_quality = minimum_base_quality
        self.minimum_alignment_coverage = minimum_alignment_coverage
        # core data
        self.reference_file = pathing(reference_file)
        self.alignment_file = pathing(alignment_file)
        self.ref_seq = get_ref_seq(reference_file)
        # to speed-up debugging
        if use_saved_pileup:
            try:
                print('=== Using Input Pileup ===')
                self.pileup = str(pathing(use_saved_pileup))
                # open it up if its a npz file
                if isinstance(self.pileup, str):
                    # self.pileup = load_npz(self.pileup)
                    self.pileup = np.load(self.pileup)
                for alignment in get_alignments(self.alignment_file):
                    self.contig_start = alignment.reference_start
                    break
            except:
                print('=== Input Pileup Failed; Building Pileup From Scratch ===')
                self.pileup = self.get_pileup()
        else:
            print('=== Building Pileup From Scratch ===')
            self.pileup = self.get_pileup()
        if save_pileup_to_destination:
            self.save_pileup(save_pileup_to_destination)
        print('=== Pilup Complete ===')

    def _meets_filter(self, total_count: int, base_count: list, ref_base: str):
        """ If read nb is different or debatable enough to be differnet (threshold) """
        if total_count < self.minimum_variant_coverage:
            return 0
        base_count = zip('ACGT', base_count)
        base_count = sorted(base_count, key = lambda x:-x[1])
        p0 = 1.0 *  base_count[0][1] / total_count
        p1 = 1.0 *  base_count[1][1] / total_count
        is_heterozygous = (p0 < 1.0 - self.heterozygous_threshold and p1 > self.heterozygous_threshold)
        is_homozygous_alt = base_count[0][0] != ref_base
        if is_heterozygous or is_homozygous_alt:   
            return 1
        return 0
    
    def _get_pileup(self):
        samfile = get_alignments(self.alignment_file)
        pileups = samfile.pileup(
            max_depth=self.minimum_alignment_coverage, 
            minimum_base_quality=self.minimum_base_quality, 
            min_mapping_quality=self.minimum_mapping_quality
        )
        for i, pileupcolumn in enumerate(pileups):
            pile = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   
            # if i < 200000: continue
            if i == 0:
                self.contig_start = pileupcolumn.reference_pos
            self.contig_end = pileupcolumn.reference_pos
            ref_base = self.ref_seq[pileupcolumn.reference_pos]
            total_count = pileupcolumn.get_num_aligned()
            pos = self.baseindex[ref_base]
            if pos != None:
                pile[pos] = total_count
            col = pileupcolumn.get_query_sequences(mark_matches=False, mark_ends=False, add_indels=True)
            for nt in col:
                if len(nt) > 1:
                    _nt, indelnts = re.split('\-|\+', nt)
                    if nt[1] == '+':
                        for indelnt in indelnts[1:]:
                            try: pos = self.baseindex[indelnt]
                            except: continue  # ignore indels larger than 9
                            if pos == None: continue
                            pile[pos+8] += 1
                else:
                    _nt = nt
                pos = self.baseindex[_nt]
                if pos == None: continue
                pile[pos+4] += 1
            call = self._meets_filter(total_count, pile[4:8], ref_base)
            if call:
                # sys.exit('YAY')
                self.variant_calls[pileupcolumn.reference_pos] = True # TODO: potential to hold info {'ref_base': ref_base, 'alt_base':}
            yield [call] + pile
            # if i == 200100: break
        samfile.close()


    def get_pileup(self):
        return np.vstack(list(self._get_pileup()))
    

    def save_pileup(self, destination):
        """
        Save scipy array to a npz file so we don't have to build the array from scratch.

        Args:
            destination ([type]): [description]
        """
        np.save(pathing(destination, new=True, overwrite=True), self.pileup)
        # save_npz(pathing(destination, new=True, overwrite=True), self.pileup)

if __name__ == '__main__':
    # pileup = Pileup(
    #     reference_file='/home/tmsincomb/Desktop/SupplementaryDataset1/SIMULATED_SNP_CONTAINING_GENOMES/neisseria/FDAARGOS_205/FDAARGOS_205_simulated.fasta',
    #     alignment_file='/home/tmsincomb/Desktop/SupplementaryDataset1/SIMULATED_SNP_CONTAINING_GENOMES/neisseria/FDAARGOS_205/dwgsim-default.mapped.bam',
    # )
    # print(pileup.pileup[:10])
    # import IPython
    # IPython.embed()
    # print(pileup.pileup[:10])

    vcf = pathing('~/Dropbox/thesis/VariantNET/testing_data/chr21/chr21.vcf')
    fasta = pathing('~/Dropbox/thesis/VariantNET/testing_data/chr21/chr21.fa')
    bam = pathing('~/Dropbox/thesis/VariantNET/testing_data/chr21/hg38.NA12878-WashU_chr21-14069662-46411975.bam')
    bami = pathing('~/Dropbox/thesis/VariantNET/testing_data/chr21/hg38.NA12878-WashU_chr21-14069662-46411975.bam.bai')
    bed = pathing('~/Dropbox/thesis/VariantNET/testing_data/chr21/CHROM21_v.3.3.2_highconf_noinconsistent.bed')
    pileup = Pileup(fasta, bam)
    print(pileup.pileup[:10])

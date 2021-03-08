"""
Scikit-Learn
- Morph df to tf
"""
import math
import multiprocessing

from Bio import SeqIO  # __init__ kicks in
import numpy as np
import pandas as pd
from pysam import AlignmentFile
from scipy.sparse import hstack, vstack, csr_matrix, save_npz, load_npz

from .tools import pathing
from .cython_numpy.cython_np_array import to_array


# TODO replace these with biopandas
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
    basepile = {
        'A':  [1, 0, 0, 0],
        'C':  [0, 1, 0, 0],
        'G':  [0, 0, 1, 0],
        'T':  [0, 0, 0, 1],
    }
    baseindex = {
        'A':  -4,
        'C':  -3,
        'G':  -2,
        'T':  -1,   
    }
    def __repr__():
        return 'Pileup 2D matrix - REF/deletion + ALT + ALT/insertion'

    def __init__(self, 
                 reference_file, 
                 alignment_file, 
                 mimimum_alignment_coverage: float = .5,
                 save_pileup_to_destination: str = None,
                 **kwargs,) -> None:
        # helper variables
        self.contig_start= None 
        self.contig_end = None
        # aligment parameters
        self.mimimum_alignment_coverage = mimimum_alignment_coverage
        # core data
        self.reference_file = pathing(reference_file)
        self.alignment_file = pathing(alignment_file)
        self.ref_seq = get_ref_seq(reference_file)
        # to speed-up debugging
        if 'pileup' in kwargs:
            self.pileup = kwargs['pileup']
            # open it up if its a npz file
            if isinstance(self.pileup, str):
                self.pileup = load_npz(self.pileup)
            for alignment in get_alignments(self.alignment_file):
                self.contig_start = alignment.reference_start
                break
        else:
            self.pileup = self.get_contig_pileup()
        if save_pileup_to_destination:
            self.save_pileup(save_pileup_to_destination)
    
    def _offset_alignments(self, a, b, left, right, dtype=np.dtype('h')):
        """
        left: push top left or push bot left
        right: push bot right
        """
        _left = np.zeros(abs(left), dtype=dtype) 
        _right = np.zeros(abs(right), dtype=dtype) 
        if np.sign(left)==-1:
            return np.vstack((a, np.append(_right, np.append(b, _left))))
        return np.vstack((np.append(a, _left), np.append(_right, b)))
    
    def _get_contig_pileup_fragment(self, alignment):
        contig_fragment = []
        for query_pos, ref_pos in alignment.get_aligned_pairs()[alignment.query_alignment_start:]: 
            if ref_pos != None:
                last_ref_pos = ref_pos # removes odd alignment edge-cases
            if ref_pos != None and query_pos != None:
                contig_fragment.extend(
                    self.basepile.get(self.ref_seq[ref_pos], [0, 0, 0, 0]) \
                    + self.basepile[alignment.query_sequence[query_pos]] \
                    + [0, 0, 0, 0])
            elif ref_pos == None and query_pos != None and contig_fragment: # BUG in query_alignment_start; it seems its not always when it starts. b
                contig_fragment[self.baseindex[alignment.query_sequence[query_pos]]] += 1
            elif ref_pos != None and query_pos == None:
                contig_fragment.extend(self.basepile.get(self.ref_seq[ref_pos], [0, 0, 0, 0]) + [0, 0, 0, 0, 0, 0, 0, 0])
            else:
                pass  # no match on either side
        return contig_fragment, last_ref_pos
                
    def _get_contig_pileup(self):
        alignmentfile = get_alignments(self.alignment_file)
        prev_reference_end = 0
        prev_contig_fragment = np.empty(1)
        for i, alignment in enumerate(alignmentfile):        
            # If alignment less that 50% aligned, it's no good and will muddy up the pileup
            skip_base, total_aln_pos = 0, 0
            for code, adv in alignment.cigartuples:
                total_aln_pos += adv
                if code == 4: 
                    skip_base += adv
            if 1.0 - 1.0 * skip_base / (total_aln_pos+1) < self.mimimum_alignment_coverage: 
                continue
            # initial start
            if i == 0:
                self.contig_start = alignment.reference_start  
                prev_reference_start = 0
                contig_pf, prev_reference_end = self._get_contig_pileup_fragment(alignment)
                prev_reference_end -= self.contig_start
                prev_contig_fragment = to_array(contig_pf)
                continue
            reference_start = alignment.reference_start - self.contig_start
            contig_pf, reference_end = self._get_contig_pileup_fragment(alignment)
            reference_end -= self.contig_start
            contig_fragment = to_array(contig_pf)
            # arrays need to to be the same size 
            left = reference_start - prev_reference_start
            right = reference_end - prev_reference_end  # can be negative offset for additional padding on incoming alignment
            left *= 12 
            right *= 12
            # padding arrays to be the same size
            merged_contig_fragment = sum(self._offset_alignments(prev_contig_fragment, contig_fragment, right, left))
            # alignments are sorted so the current reference slice is next
            prev_contig_fragment = merged_contig_fragment[(reference_start - prev_reference_start)*12:]
            # previous reference start to current reference start slice for vstack
            fragment = merged_contig_fragment[:(reference_start - prev_reference_start)*12]
            # update positions
            prev_reference_end = reference_end if reference_end >= prev_reference_end else prev_reference_end
            prev_reference_start = reference_start
            if fragment.size!=0: # repeating alignments at same ref start position
                yield csr_matrix(fragment.reshape(-1, 12), dtype=np.dtype('h'))
        self.contig_end = prev_reference_end + self.contig_start
        if prev_contig_fragment.size!=0:
            yield csr_matrix(prev_contig_fragment.reshape(-1, 12), dtype=np.dtype('h'))        
        alignmentfile.close()  # todo: do i need this?

    def get_contig_pileup(self):
        return vstack(self._get_contig_pileup())

    def save_pileup(self, destination):
        save_npz(pathing(destination, new=True), self.pileup)

# TODO: add kwargs to each function
# TODO: update kwargs pileup?
class Tensors:

    def __init__(self, cpileup, **kwargs) -> None:
        self.pileup = cpileup
        self.tensors = [self.supports_reference_with_deletions(), 
                        self.supports_alt_with_insertions(), 
                        self.supports_alt()]

    def supports_reference_with_deletions(self, index_ref_with_del: tuple = (0, 4)):
        return self.pileup[:, slice(*index_ref_with_del)]

    def supports_alt_with_insertions(self, 
                                     index_ref_with_del: tuple = (0, 4), 
                                     index_alt: tuple = (4, 8), 
                                     index_alt_inserts: tuple = (8, 12)):
        return (self.pileup[:, slice(*index_alt_inserts)] + self.pileup[:, slice(*index_alt)]) - self.pileup[:, slice(*index_ref_with_del)] 

    def supports_alt(self, index_ref_with_del: tuple = (0, 4), index_alt: tuple = (4, 8)):
        return self.pileup[:, slice(*index_alt)] - self.pileup[:, slice(*index_ref_with_del)]


class VariantCaller:

    def __init__(self, 
                 ref_seq, 
                 alt_pileup, 
                 contig_start, 
                 minimum_coverage: int = 10, 
                 heterozygous_threshold: float = .15, 
                 minimum_variant_radius: int = 12, 
                 **kwargs) -> None:
        self.alt_pileup = alt_pileup
        self.ref_seq = ref_seq
        self.contig_start = contig_start
        self.minimum_coverage = minimum_coverage
        self.heterozygous_threshold = heterozygous_threshold  # minimum difference a secondary alt needs to be for a variant to be called. 
        self.minimum_variant_radius = minimum_variant_radius  # distance variants are allowed to be without being classified as complex

    def _meets_filter(self, total_count: int, base_count: list, ref_base: str):
        """ If read nb is different or debatable enoough to be differnet (threshold) """
        base_count = zip('ACGT', base_count)
        base_count = sorted(base_count, key = lambda x:-x[1])
        if base_count[0][0] != ref_base:
            return base_count[0][0]
        p0 = 1.0 *  base_count[0][1] / total_count
        p1 = 1.0 *  base_count[1][1] / total_count
        if (p0 < 1.0 - self.heterozygous_threshold and p1 > self.heterozygous_threshold):   
            return base_count[0][0]
        return False

    def get_variant_calls(self):
        batch_start = 0
        batch_size = 10000  # no more speedup past 10K; most likely memory limitations
        limit = None  # for dubugging only
        segment_end = limit or self.alt_pileup.shape[0]
        variant_calls = {}
        a = self.alt_pileup[:, :].sum(axis=1, dtype=np.dtype('h')).flatten()
        b = np.where(a >= self.minimum_coverage, a, 0)[0]  # TODO: this can be used as a filter to avoid a zip.
        for offset in range(batch_start, segment_end, batch_size):
            ref_start =  self.contig_start + offset
            position = ref_start
            for total_count, row, ref_base in zip(b[offset:offset+batch_size].tolist(), 
                                                  self.alt_pileup[offset:offset+batch_size, :].toarray(), 
                                                  self.ref_seq[ref_start:ref_start+batch_size]):
                position += 1
                if total_count == 0:
                    continue
                alt_base = self._meets_filter(total_count, row, ref_base)
                if alt_base != False:
                    variant_calls[position]={
                        'total_count': total_count, 
                        'row': row, 
                        'ref_base': ref_base, 
                        'alt_base': alt_base,
                        'fall_back_genotype': 6,  # non-variant by default if not in the "truth" VCF
                    }
        # updateing the fall_back_genotype to 7 for complex if the variants are too close to eachother
        Y_pos = sorted(variant_calls)
        cpos = Y_pos[0]
        for pos in Y_pos[1:]:
            if abs(pos - cpos) < self.minimum_variant_radius:
                variant_calls[pos]['fall_back_genotype'] = 7
            cpos = pos
        return variant_calls


class Preprocess(Pileup, Tensors, VariantCaller):

    def __init__(self, 
                 reference_file: str, 
                 alignment_file: str, 
                 vcf_file: str,
                 bed_file: str = None,
                 window_size: int = 15,
                 index_alt: tuple = (4, 8), 
                 **kwargs) -> None:
        self.vcf_file = pathing(vcf_file)
        self.bed_file = pathing(bed_file) if bed_file else None
        self.window_size = window_size
        Pileup.__init__(self, reference_file, alignment_file, **kwargs)
        Tensors.__init__(self, self.pileup, **kwargs)
        VariantCaller.__init__(self, self.ref_seq, self.pileup[:, slice(*index_alt)], self.contig_start, **kwargs)
        self.variant_calls = self.get_variant_calls()

    def __repr__():
        return "Preprocessing!"

    def get_training_array(self):
        """ VCF for training is considered absolute truth """
        y_index = {
            'A': 0, 
            'C': 1,
            'G': 2, 
            'T': 3,
            '0/1': 4, '1/0': 4, # heterozygous
            '1/1': 5, # homozygous
            '0/0': 6, # non-variant :: assigned where alignments are not found to be variants. Need to finish populating with bam file.
            # 7 :: complex/non-snp :: assigned to be a variant that is an indel, but not an SNP
        }
        y = [0, 0, 0, 0, 0, 0, 0, 0] # ['A', 'C', 'T', 'G', het, hom, non, complex]
        Y = {}
        if self.bed_file:
            focus_regions = pd.read_csv(self.bed_file, delimiter='\t', header=None)[[1, 2]].apply(tuple, axis=1).tolist()
            focus_regions = pd.arrays.IntervalArray.from_tuples(focus_regions, closed='both')
        vcf = pd.read_vcf(self.vcf_file)  # Should only have one sample
        if len(vcf.columns) > 10:
            exit(f'ERROR :: VCF file has too many samples')
        if not self.window_size % 2: print('shit man, the window needs to be odd; needs to have a middle position')
        left_offset = math.floor(self.window_size / 2)
        right_offset = math.ceil(self.window_size / 2)
        for row in vcf.itertuples():
            y_vec = y[:] # ['A', 'C', 'T', 'G', het, hom, non, complex]
            if self.bed_file: 
                if not any(focus_regions.contains(row.POS)): 
                    continue
            # get genotype call. default to non-variant
            genotype = row[-1]['GT'].replace('|', '/')
            genotype_index = y_index.get(genotype, 7)
            if len(row.REF) > 1 or len(row.ALT) > 1:
                genotype_index = 7
            # HETEROZYGOUS
            if genotype_index == 4:
                y_vec[y_index[row.REF[0]]] += .5
                y_vec[y_index[row.ALT[0]]] += .5
            # HOMOZYGOUS
            elif genotype_index == 5:
                y_vec[y_index[row.ALT[0]]] += 1
            # NON-VARIANT
            elif genotype_index == 6:
                y_vec[y_index[row.REF[0]]] += 1
            # COMPLEX
            elif genotype_index == 7:
                # todo: this shouldnt be always in favor of alt
                y_vec[y_index[row.ALT[0]]] += 1 # todo: maybe take avgs if this messes with the output
            y_vec[genotype_index] = 1
            Y[row.POS] = y_vec        
        X_initial = []
        Y_initial = []
        position_array = []
        for position in Y:
            if self.bed_file: 
                if not any(focus_regions.contains(position)): 
                    continue
            tp = position - self.contig_start - 1
            if tp < 0:
                continue
            tensor_stack = np.stack([tensor[tp-left_offset:tp+right_offset].toarray() for tensor in self.tensors], axis=2)
            if tensor_stack.size == 0:
                print(position, 'tp end')
                break
            position_array.append(position)
            X_initial.append(tensor_stack)
            Y_initial.append(Y[position])
        false_positives = sorted(set(self.variant_calls)- set(Y))
        for position in false_positives:
            if self.bed_file: 
                if not any(focus_regions.contains(position)): 
                    continue
            y = [0, 0, 0, 0, 0, 0, 1, 0]
            base_position = y_index.get(self.variant_calls[position]['ref_base'])
            p = position - self.contig_start - 1
            if base_position:
                tensor_stack = np.stack([tensor[p-left_offset:p+right_offset].toarray() for tensor in self.tensors], axis=2)
                if tensor_stack.size == 0:
                    print(position, 'shit -> fp end; should not happen')
                    break
                y[base_position] += 1v
                position_array.append(position)
                Y_initial.append(y)  # like this incase we want to modify the base 
                X_initial.append(tensor_stack)
        Xarray = np.stack(X_initial)
        Yarray = np.stack(Y_initial)
        return Xarray, Yarray, position_array # Xarray, Yarray
            
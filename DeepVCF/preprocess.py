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
from .pileup import Pileup


# TODO replace these with biopandas
def get_ref_seq(reference_file: str, format: str = 'fasta'):
    reference_file = pathing(reference_file)
    seqrecords = list(SeqIO.parse(reference_file, format=format))
    if len(seqrecords) > 1:
        exit('ERROR :: Reference fasta must be a single complete sequence')
    return seqrecords[0].seq


def get_alignments(alignment_file, threads=multiprocessing.cpu_count()):
    return AlignmentFile(pathing(alignment_file), threads=threads)


# class Pileup:
#     """ """
#     basepile = {
#         'A':  [1, 0, 0, 0],
#         'C':  [0, 1, 0, 0],
#         'G':  [0, 0, 1, 0],
#         'T':  [0, 0, 0, 1],
#     }
#     baseindex = {
#         'A':  -4,
#         'C':  -3,
#         'G':  -2,
#         'T':  -1,   
#     }
#     def __repr__():
#         return 'Pileup 2D matrix - REF/deletion + ALT + ALT/insertion'

#     def __init__(self, 
#                  reference_file, 
#                  alignment_file, 
#                  mimimum_alignment_coverage: float = .75,
#                  save_pileup_to_destination: str = None,
#                  **kwargs,) -> None:
#         # helper variables
#         self.contig_start= None 
#         self.contig_end = None
#         self.mapping_coverages=[]
#         # aligment parameters
#         self.mimimum_alignment_coverage = mimimum_alignment_coverage
#         # core data
#         self.reference_file = pathing(reference_file)
#         self.alignment_file = pathing(alignment_file)
#         self.ref_seq = get_ref_seq(reference_file)
#         # to speed-up debugging
#         if 'pileup' in kwargs:
#             try:
#                 print('=== Using Input Pileup ===')
#                 self.pileup = str(pathing(kwargs['pileup']))
#                 # open it up if its a npz file
#                 if isinstance(self.pileup, str):
#                     self.pileup = load_npz(self.pileup)
#                 for alignment in get_alignments(self.alignment_file):
#                     self.contig_start = alignment.reference_start
#                     break
#             except:
#                 print('=== Input Pileup Failed; Building Pileup From Scratch ===')
#                 self.pileup = self.get_contig_pileup()
#         else:
#             print('=== Building Pileup From Scratch ===')
#             self.pileup = self.get_contig_pileup()
#         if save_pileup_to_destination:
#             self.save_pileup(save_pileup_to_destination)
#         print('=== Pilup Complete ===')
        
#     def _offset_alignments(self, a, b, left, right, dtype=np.dtype('h')):
#         """
#         left: push top left or push bot left
#         right: push bot right
#         """
#         _left = np.zeros(abs(left), dtype=dtype) 
#         _right = np.zeros(abs(right), dtype=dtype) 
#         if np.sign(left)==-1:
#             return np.vstack((a, np.append(_right, np.append(b, _left))))
#         return np.vstack((np.append(a, _left), np.append(_right, b)))
    
#     def _get_contig_pileup_fragment(self, alignment):
#         contig_fragment = []
#         for query_pos, ref_pos in alignment.get_aligned_pairs()[alignment.query_alignment_start:alignment.query_alignment_end]: 
#             if ref_pos != None:
#                 last_ref_pos = ref_pos # removes odd alignment edge-cases
#             if ref_pos != None and query_pos != None:
#                 contig_fragment.extend(
#                     self.basepile.get(self.ref_seq[ref_pos], [0, 0, 0, 0]) \
#                     + self.basepile[alignment.query_sequence[query_pos]] \
#                     + [0, 0, 0, 0])
#             elif ref_pos == None and query_pos != None and contig_fragment: # BUG in query_alignment_start; it seems its not always when it starts. b
#                 contig_fragment[self.baseindex[alignment.query_sequence[query_pos]]] += 1
#             elif ref_pos != None and query_pos == None:
#                 contig_fragment.extend(self.basepile.get(self.ref_seq[ref_pos], [0, 0, 0, 0]) + [0, 0, 0, 0, 0, 0, 0, 0])
#             else:
#                 pass  # no match on either side
#         return contig_fragment, last_ref_pos
                
#     def _get_contig_pileup(self):
#         alignmentfile = get_alignments(self.alignment_file)
#         prev_reference_end = 0
#         prev_contig_fragment = np.empty(1)
#         for i, alignment in enumerate(alignmentfile):        
#             # If alignment less that 50% aligned, it's no good and will muddy up the pileup
#             skip_base, total_aln_pos = 0, 0
#             for code, adv in alignment.cigartuples:
#                 total_aln_pos += adv
#                 if code == 4: 
#                     skip_base += adv
#             mapping_coverage = 1.0 - 1.0 * skip_base / (total_aln_pos+1)
#             if mapping_coverage < self.mimimum_alignment_coverage: 
#                 continue
#             self.mapping_coverages.append(mapping_coverage)
#             # initial start
#             if i == 0 or not self.contig_start:
#                 self.contig_start = alignment.reference_start  
#                 prev_reference_start = 0
#                 contig_pf, prev_reference_end = self._get_contig_pileup_fragment(alignment)
#                 prev_reference_end -= self.contig_start
#                 prev_contig_fragment = to_array(contig_pf)
#                 continue
#             reference_start = alignment.reference_start - self.contig_start
#             contig_pf, reference_end = self._get_contig_pileup_fragment(alignment)
#             reference_end -= self.contig_start
#             contig_fragment = to_array(contig_pf)
#             # arrays need to to be the same size 
#             left = reference_start - prev_reference_start
#             right = reference_end - prev_reference_end  # can be negative offset for additional padding on incoming alignment
#             left *= 12 
#             right *= 12
#             # padding arrays to be the same size
#             merged_contig_fragment = sum(self._offset_alignments(prev_contig_fragment, contig_fragment, right, left))
#             # alignments are sorted so the current reference slice is next
#             prev_contig_fragment = merged_contig_fragment[(reference_start - prev_reference_start)*12:]
#             # previous reference start to current reference start slice for vstack
#             fragment = merged_contig_fragment[:(reference_start - prev_reference_start)*12]
#             # update positions
#             prev_reference_end = reference_end if reference_end >= prev_reference_end else prev_reference_end
#             prev_reference_start = reference_start
#             if fragment.size!=0: # repeating alignments at same ref start position
#                 yield csr_matrix(fragment.reshape(-1, 12), dtype=np.dtype('h'))
#         self.contig_end = prev_reference_end + self.contig_start
#         if prev_contig_fragment.size!=0:
#             yield csr_matrix(prev_contig_fragment.reshape(-1, 12), dtype=np.dtype('h'))        
#         alignmentfile.close()  # todo: do i need this?

#     #  TODO: There is a bottleneck somewhere in here. Memory issues are most likely the issue since time is linear with matrix depth & time allocating the actual memory was solved with cython.
#     def get_contig_pileup(self):
#         """
#         Get pileup of reference+deletions, alt and alt+insertions for dynamic creation of tensors.

#         Returns:
#             csr_matrix: sparce column matrix to hold the complete segment of sequences in a tiny amount of memory.
#         """
#         return vstack(self._get_contig_pileup())

#     def save_pileup(self, destination):
#         """
#         Save scipy array to a npz file so we don't have to build the array from scratch.

#         Args:
#             destination ([type]): [description]
#         """
#         save_npz(pathing(destination, new=True, overwrite=True), self.pileup)

# TODO: add kwargs to each function
class Tensors:

    def __init__(self, cpileup, **kwargs) -> None:
        self.pileup = cpileup
        self.tensors = [self.supports_reference_with_deletions(), 
                        # self.supports_alt(),
                        self.supports_alt_diff(), 
                        self.supports_alt_diff_with_insertions()]
        print('=== Tensors Complete ===')

    def supports_reference_with_deletions(self, index_ref_with_del: tuple = (1, 5)):
        return self.pileup[:, slice(*index_ref_with_del)]


    def supports_alt(self, index_alt: tuple = (5, 9)):
        return self.pileup[:, slice(*index_alt)]

    def supports_alt_diff_with_insertions(self, 
                                          index_ref_with_del: tuple = (1, 5), 
                                          index_alt: tuple = (5, 9), 
                                          index_alt_inserts: tuple = (9, 13)):
        return (self.pileup[:, slice(*index_alt_inserts)] + self.pileup[:, slice(*index_alt)]) - self.pileup[:, slice(*index_ref_with_del)] 

    def supports_alt_diff(self, index_ref_with_del: tuple = (1, 5), index_alt: tuple = (5, 9)):
        return self.pileup[:, slice(*index_alt)] - self.pileup[:, slice(*index_ref_with_del)]


# class VariantCaller:

#     def __init__(self, 
#                  ref_seq, 
#                  alt_pileup, 
#                  contig_start, 
#                  minimum_coverage: int = 20, 
#                  heterozygous_threshold: float = .25, 
#                  minimum_variant_radius: int = 12, 
#                  **kwargs) -> None:
#         self.alt_pileup = alt_pileup
#         self.ref_seq = ref_seq
#         self.contig_start = contig_start
#         self.minimum_coverage = minimum_coverage
#         self.heterozygous_threshold = heterozygous_threshold  # minimum difference a secondary alt needs to be for a variant to be called. 
#         self.minimum_variant_radius = minimum_variant_radius  # distance variants are allowed to be without being classified as complex

#     def _meets_filter(self, total_count: int, base_count: list, ref_base: str):
#         """ If read nb is different or debatable enoough to be differnet (threshold) """
#         base_count = zip('ACGT', base_count)
#         base_count = sorted(base_count, key = lambda x:-x[1])
#         if base_count[0][0] != ref_base:
#             return base_count[0][0]
#         p0 = 1.0 *  base_count[0][1] / total_count
#         p1 = 1.0 *  base_count[1][1] / total_count
#         if (p0 < 1.0 - self.heterozygous_threshold and p1 > self.heterozygous_threshold):   
#             return base_count[0][0]
#         return False

#     def get_variant_calls(self):
#         batch_start = 0
#         batch_size = 10000  # no more speedup past 10K; most likely memory limitations
#         limit = None  # for dubugging only
#         segment_end = limit or self.alt_pileup.shape[0]
#         variant_calls = {}
#         a = self.alt_pileup[:, :].sum(axis=1, dtype=np.dtype('h')).flatten()
#         b = np.where(a >= self.minimum_coverage, a, 0)[0]  # TODO: this can be used as a filter to avoid a zip.
#         for offset in range(batch_start, segment_end, batch_size):
#             ref_start =  self.contig_start + offset
#             position = ref_start
#             for total_count, row, ref_base in zip(b[offset:offset+batch_size].tolist(), 
#                                                   self.alt_pileup[offset:offset+batch_size, :], #.toarray(), 
#                                                   self.ref_seq[ref_start:ref_start+batch_size]):
#                 position += 1
#                 if total_count == 0:
#                     continue
#                 alt_base = self._meets_filter(total_count, row, ref_base)
#                 if alt_base != False:
#                     variant_calls[position]={
#                         'total_count': total_count, 
#                         'row': row, 
#                         'ref_base': ref_base, 
#                         'alt_base': alt_base,
#                         'fall_back_genotype': 6,  # non-variant by default if not in the "truth" VCF
#                     }
#         # Variants too close to eachother cause bad data. Best to ignore them completely.
#         Y_pos = sorted(variant_calls)
#         cpos = Y_pos[0]
#         for pos in Y_pos[1:]:
#             if abs(pos - cpos) < self.minimum_variant_radius:
#                 del variant_calls[pos]
#             cpos = pos
#         return variant_calls


class Preprocess(Pileup, Tensors):

    def __init__(self, 
                 reference_file: str, 
                 alignment_file: str, 
                 vcf_file: str = None,
                 bed_file: str = None,
                 window_size: int = 15,
                #  index_alt: tuple = (4, 8),
                #  variant_calls: str = None,
                #  save_variant_calls_to_destination: str = None,
                 **kwargs) -> None:
        self.vcf_file = pathing(vcf_file) if vcf_file else None
        self.bed_file = pathing(bed_file) if bed_file else None
        self.window_size = window_size
        Pileup.__init__(self, reference_file, alignment_file, **kwargs)
        #self.variant_calls = variant_calls or {v+self.contig_start + 1:True for v in np.nonzero(self.pileup[:, :1])[0]}  # TODO: where to move this?
        Tensors.__init__(self, self.pileup, **kwargs)
        # This Variant Caller bad/insensitive on purpose. We need a wide net to catch the real variants while also allowing us to skip windows where variants are less likely.
        # VariantCaller.__init__(self, self.ref_seq, self.pileup[:, slice(*index_alt)], self.contig_start, **kwargs)
        # TODO: clean this up 
        # if variant_calls:
        #     try:
        #         print('=== Using Stored Variant Calls ===')
        #         self.variant_calls = pd.read_csv(pathing(variant_calls), index_col=0).to_dict('index')
        #     except:
        #         print('=== Variant Calls Stored File Faile; Building Variant Calls From Scratch ===')
        #         self.variant_calls = self.get_variant_calls()
        # else:
        #     print('=== Building Variant Calls From Scratch ===')
        #     self.variant_calls = self.get_variant_calls()
        # if save_variant_calls_to_destination:
        #     # TODO: add check if parent path exists
        #     variant_calls_destination = pathing(save_variant_calls_to_destination, new=True, overwrite=True)
        #     pd.DataFrame.from_dict(self.variant_calls, orient='index').to_csv(variant_calls_destination, index=True)
        # print('=== Variant Calls Complete ===')

    def __repr__():
        return "Preprocessing!"

    def get_training_array(self, window_size: str = None):
        """ VCF for training is considered absolute truth """
        self.window_size = window_size or self.window_size
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
        y = [0, 0, 0, 0, 0, 0, 0, 0] # ['A', 'C', 'T', 'G', hom-ref, het, hom-alt, complex-dump]
        Y = {}
        X_initial = []
        Y_initial = []
        position_array = []
        left_offset = math.floor(self.window_size / 2)
        right_offset = math.ceil(self.window_size / 2)
        if not self.window_size % 2: print('shit man, the window needs to be odd; needs to have a middle position')
        if self.bed_file:
            focus_regions = pd.read_csv(self.bed_file, delimiter='\t', header=None)[[1, 2]].apply(tuple, axis=1).tolist()
            focus_regions = pd.arrays.IntervalArray.from_tuples(focus_regions, closed='both')
        count = 0
        too_complex = set()
        if self.vcf_file:
            vcf = pd.read_vcf(self.vcf_file)  # Should only have one sample
            if len(vcf.columns) > 10:
                exit(f'ERROR :: VCF file has too many samples')
            vpos = -float('inf')
            for row in vcf.itertuples():
                y_vec = y[:]  # ['A', 'C', 'T', 'G', het, hom, non, complex]
                # if self.bed_file: 
                #     if not any(focus_regions.contains(row.POS-1)):  # bed file 0-index
                #         count += 1
                #         continue
                # get genotype call. default to non-variant
                genotype = row[-1]['GT'].replace('|', '/')
                genotype_index = y_index.get(genotype, 7)
                # HETEROZYGOUS
                if genotype_index == 4:
                    y_vec[y_index[row.REF[0]]] = .5
                    y_vec[y_index[row.ALT[0]]] = .5
                    # y_vec[y_index[row.REF[0]]] = 1
                    # y_vec[y_index[row.ALT[0]]] = 1
                # HOMOZYGOUS
                elif genotype_index == 5:
                    y_vec[y_index[row.ALT[0]]] = 1
                    # y_vec[y_index[row.ALT[0]]] = 1
                # NON-VARIANT
                elif genotype_index == 6:
                    y_vec[y_index[row.REF[0]]] = 1
                    # y_vec[y_index[row.REF[0]]] = 1
                # COMPLEX
                elif genotype_index == 7:
                    # todo: this shouldnt be always in favor of alt
                    y_vec[y_index[row.ALT[0]]] = 1 # todo: maybe take avgs if this messes with the output
                # makes sure we get the proper het base call before changing the gt to complex.
                if len(row.REF) > 1 or len(row.ALT) > 1:
                    genotype_index = 7
                if abs(row.POS - vpos) < self.minimum_variant_radius:
                    genotype_index = 7
                    try:
                        Y[vpos][4] = 0
                        Y[vpos][5] = 0
                        Y[vpos][6] = 0
                        Y[vpos][7] = 1
                    except:
                        pass
                # if len(row.REF) > 5 or len(row.ALT) > 5:
                #     too_complex.add(row.POS)
                #     vpos = row.POS
                #     continue
                vpos = row.POS
                y_vec[genotype_index] = 1
                Y[row.POS] = y_vec   
            for position in Y:
                # if self.bed_file: 
                #     if not any(focus_regions.contains(position-1)):  # bed file 0-index
                #         continue
                tp = position - self.contig_start - 1
                if tp < 0:  # calls before contig :: incase a bed file was used 
                    continue
                tensor_stack = np.stack([tensor[tp-left_offset:tp+right_offset] for tensor in self.tensors], axis=2)
                if tensor_stack.size == 0:  # calls after contig :: incase a bed file was used
                    break 
                position_array.append(position)
                X_initial.append(tensor_stack)
                Y_initial.append(Y[position])
        # print('focus count', count)
        # false_positives = sorted(set(self.variant_calls) - (set(Y) | too_complex))
        # self.false_positives = false_positives
        # ref_seq_seg = self.ref_seq[self.contig_start-1:self.contig_end]
        # print('false-p', len(false_positives))
        # for position in false_positives[:]:
        else:
            for position in self.variant_calls:
                p = position - self.contig_start - 1  # numpy array 0-index
                if self.bed_file: 
                    if not any(focus_regions.contains(position-1)):  # bed file 0-index 
                        continue
                y = [0, 0, 0, 0, 0, 0, 1, 0]
                # base_position = y_index.get(self.variant_calls[position]['ref_base'])
                base_position = y_index.get(self.ref_seq[position-1])  # bypthon 0-index
                # p = position + self.contig_start
                if base_position:
                    if p - left_offset < 0:  # TODO: circularize if for plasmids
                        continue
                    tensor_stack = np.stack([tensor[p-left_offset:p+right_offset] for tensor in self.tensors], axis=2)
                    vec = np.transpose(tensor_stack, axes=(0,2,1))
                    if sum(vec[7,:,0]) < 5:
                        continue
                        # print(position)
                    if tensor_stack.size == 0:
                        print(position, 'shit -> fp end; should not happen')
                        break
                    y[base_position] = 1
                    position_array.append(position)
                    Y_initial.append(y)  # like this incase we want to modify the base 
                    X_initial.append(tensor_stack)

        print(position)
        Xarray = np.stack(X_initial).astype('float64')
        Yarray = np.stack(Y_initial).astype('float64')
        return Xarray, Yarray, position_array # Xarray, Yarray
            

    def get_window(self, position, window_size=15):
        position = position - self.contig_start - 1
        if not window_size % 2: print('shit man, the window needs to be odd; needs to have a middle position')
        left_offset = math.floor(window_size / 2)
        right_offset = math.ceil(window_size / 2)
        return np.stack([tensor[position-left_offset:position+right_offset] for tensor in self.tensors], axis=2)

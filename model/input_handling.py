from pyfaidx import Fasta
from BCBio import GFF
import pprint
from Bio import SeqFeature, SeqRecord
import numpy as np
import pandas as pd
import pickle
import math

from typing import Optional, Tuple, List


# todo:
# read in gff
# find genes start / end
# calculate start / end of flanking regions and directly extract OR
# calculate start / end for later extraction
# combine all extracted sequences into one fasta file

CENTRAL_PADDING = 20

def onehot(seq):
    code = {'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            'unk': [0, 0, 0, 0]}
    encoded = np.zeros(shape=(len(seq), 4))
    for i, nt in enumerate(seq):
        if nt in ['A', 'C', 'G', 'T']:
            encoded[i, :] = code[nt]
        else:
            encoded[i, :] = code['unk']
    return encoded


def find_genes(gff_path: str) -> pd.DataFrame:
    chromosomes, starts, ends, strands, gene_ids = [], [], [], [], []
    with open(gff_path, "r") as file:
        filter = dict(gff_type=["gene"])
        rec: SeqRecord.SeqRecord
        for rec in GFF.parse(file, limit_info=filter):
            feat: SeqFeature.SeqFeature
            for feat in rec.features:
                # extracted_seq = feat.extract(fasta_obj[rec.id])
                strand = feat.location.strand #type:ignore
                if strand not in [-1, 1]:
                    print(f"no proper strand available for {feat.id} on {rec.id}")
                    # nothing should be appended, if one of the columns cant be filled properly. 
                    continue
                strand = "+" if strand == 1 else "-"
                chromosomes.append(rec.id)
                starts.append(feat.location.start) #type:ignore
                ends.append(feat.location.end) #type:ignore
                strands.append(strand)
                gene_ids.append(feat.id)
                break

    return pd.DataFrame(data={
        "chromosome": chromosomes,
        "start": starts,
        "end": ends,
        "strand": strands,
        "gene_id": gene_ids
    })


def find_start_end(start: int, end: int, intragenic: int, extragenic: int, strand: str) -> Tuple[int, int, int, int, int]:
    # if gene length is less than 1000, the intragenic regions from TTS and TSS will overlap
    gene_length = abs(end - start)
    if gene_length < 2 * intragenic:
        # adjust length to be extracted so it does not overlap, and add padding to the center,
        # so that the extracted sequence maintains the same length and the position of the TSS
        # and TTS remain.
        longer_intra_gene = math.ceil(gene_length / 2)
        shorter_intra_gene = math.floor(gene_length / 2)
        additional_padding = 2 * intragenic - gene_length
    else:
        longer_intra_gene = shorter_intra_gene = intragenic
        additional_padding = 0

    if strand == '+':
        prom_start, prom_end = start - extragenic, start + longer_intra_gene
        term_start, term_end = end - shorter_intra_gene, end + extragenic

    else:
        prom_start, prom_end = end - longer_intra_gene, end + extragenic
        term_start, term_end = start - extragenic, start + shorter_intra_gene
    return prom_start, prom_end, term_start, term_end, additional_padding


def extract_seq(fasta: Fasta, genes: pd.DataFrame, intragenic: int = 500, extragenic: int = 1000) -> Tuple[List[np.ndarray], List[str]]:
    encoded_train_seqs, train_ids = [], []
    lists = encoded_train_seqs, train_ids
    for chrom, start, end, strand, gene_id in genes.values:
        vals = find_start_end(start=start, end=end, intragenic=intragenic, extragenic=extragenic, strand=strand)
        prom_start, prom_end, term_start, term_end, additional_padding = vals
        append_sequences(prom_start=prom_start, prom_end=prom_end, term_start=term_start, term_end=term_end,
                            central_padding=CENTRAL_PADDING, additional_padding=additional_padding, chrom=chrom,
                            gene_id=gene_id, lists=lists, fasta=fasta, intragenic=intragenic, extragenic=extragenic)

    return encoded_train_seqs, train_ids


def append_sequences(prom_start, prom_end, term_start, term_end, central_padding, additional_padding,
                        lists, chrom, gene_id, fasta, intragenic, extragenic) -> None:
    encoded_train_seqs, train_ids = lists
    if prom_start > 0 and term_start > 0:
        if prom_start < prom_end and term_start < term_end:
            direction = 1
        elif prom_start > prom_end and term_start > term_end:
            direction = -1
        encoded_seq = np.concatenate([onehot(fasta[chrom][prom_start:prom_end])[::direction, ::direction],
                                                np.zeros(shape=(central_padding + additional_padding, 4)),
                                                onehot(fasta[chrom][term_start:term_end])[::direction, ::direction]])

        if encoded_seq.shape[0] == 2*(extragenic + intragenic) + central_padding:
            encoded_train_seqs.append(encoded_seq)
            train_ids.append(gene_id)


def extract_string(fasta_obj: Fasta, gene_df: pd.DataFrame, intragenic: int, extragenic: int, output_file: str, central_padding: int):
    genes_seqs = {}
    for chrom, start, end, strand, gene_id in gene_df.values:
        vals = find_start_end(start=start, end=end, intragenic=intragenic, extragenic=extragenic, strand=strand)
        prom_start, prom_end, term_start, term_end, additional_padding = vals
        if prom_start > 0 and term_start > 0:
            if prom_start < prom_end and term_start < term_end:
                direction = 1
            elif prom_start > prom_end and term_start > term_end:
                direction = -1
            else:
                raise  ValueError("Issue..")
            promoter = fasta_obj[chrom][prom_start:prom_end:direction]
            terminator = fasta_obj[chrom][term_start:term_end:direction]
            padding = (central_padding + additional_padding) * "N"
            gene_flanks = "".join(promoter) + padding + "".join(terminator)
            genes_seqs[f"{chrom}_{gene_id}_{extragenic}_{intragenic}"] = gene_flanks
    
    with open(output_file, "w") as f:
        for id, sequence in genes_seqs.items():
            f.write(f">{id}\n{sequence}\n")




def extract_gene_flanking_regions(fasta_path: str, gff_path: str, output_path: str, extract_one_hot_flag: bool, extract_string_flag: bool, extragenic: int, intragenic: int) -> Optional[Tuple[List, List]]:
    if not (extract_one_hot_flag or extract_string_flag):
        raise  ValueError("either extraction as string or one hot encoded np array is necessary!")
    fasta_obj = Fasta(fasta_path, as_raw=True, sequence_always_upper=True, read_ahead=10000)
    gene_df = find_genes(gff_path)
    if extract_string_flag:
        extract_string(fasta_obj, gene_df, intragenic=intragenic, extragenic=extragenic, output_file=output_path, central_padding=CENTRAL_PADDING)
    if extract_one_hot_flag:
        sequences, gene_ids = extract_seq(fasta=fasta_obj, genes=gene_df, intragenic=intragenic, extragenic=extragenic)
        return sequences, gene_ids


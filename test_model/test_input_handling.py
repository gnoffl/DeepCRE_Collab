from pyfaidx import Fasta
from BCBio import GFF
import pprint
from Bio import SeqFeature, SeqRecord
import numpy as np
import pandas as pd
import pickle
import math

from typing import Tuple, List

import extract_sequences

# test empty sequence
# test names being found in output
# test sequences being correct
    # short gene
    # eactly right length
    # too long
# one chromosome vs many
# different data formats (of fasta, and then gff/gtf/...)



def test_fasta():
    fasta_obj = Fasta("/home/gernot/Downloads/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa", as_raw=True, sequence_always_upper=True, read_ahead=10000)
    print(fasta_obj.keys())
    print(fasta_obj["1"])


if __name__ == "__main__":
    extract_sequences.extract_gene_flanking_regions(fasta_path="/home/gernot/Downloads/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa", gff_path="/home/gernot/Downloads/Arabidopsis_thaliana.TAIR10.59.gff3", output_path="test.fa", extract_one_hot_flag=False, extract_string_flag=True, extragenic=100, intragenic=50)

    # test_fasta()
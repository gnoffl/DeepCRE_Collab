from pyfaidx import Fasta
from BCBio import GFF
import pprint
from Bio import SeqFeature, SeqRecord
import numpy as np
import pandas as pd
import pickle
import math
import os
import sys

from typing import Tuple, List
model_path = os.path.join(os.path.dirname(__file__), "..", "model")
sys.path.insert(1, model_path)
from utils import prepare_valid_seqs
import input_handling as ih

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


def write_mockup_data():
    sequence = "A" * 500 + "C" * 500 + "T" * 500 + "G" * 500
    sequence = sequence * 5
    name = "1 mock_up_sequence"
    fasta_path = os.path.join(os.path.dirname(__file__), "test_data", "test_input.fa")
    with open(fasta_path, "w") as f:
        f.write(f">{name}\n")
        for i in range(0, len(sequence) + 1, 80):
            f.write(f"{sequence[i:i+80]}\n")
        if i != len(sequence):
            f.write(f"{sequence[i+80:-1]}\n")
    


if __name__ == "__main__":
    #todo: 
    #   create simple test sequence + annotation
    #   make sure this works with string as well as one hot extraction
    #   add argument parser to extract stuff from anywhere, if necessary??
    # ih.extract_gene_flanking_regions(fasta_path="/home/gernot/Downloads/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa",
    #                                  gff_path="/home/gernot/Downloads/Arabidopsis_thaliana.TAIR10.59.gff3",
    #                                  output_path="test.fa", extract_one_hot_flag=False, extract_string_flag=True, 
    #                                  extragenic=100, intragenic=50)
    write_mockup_data()


    # test_fasta()
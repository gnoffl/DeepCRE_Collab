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

import unittest

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


class TestExtractor(unittest.TestCase):
    encoding = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'unk': [0, 0, 0, 0],
        'N': [0, 0, 0, 0]
    }

    def test_string_extraction(self):
        test_data_path = os.path.join(os.path.dirname(__file__), "test_data")
        fasta_path = os.path.join(test_data_path, "test_input_short.fa")
        gff_path = os.path.join(test_data_path, "test_input_short.gff3")
        output_path = os.path.join(test_data_path, "test_result_short.txt")
        write_mockup_data(repeat_length=10, file_name="test_input_short.fa")
        ih.extract_gene_flanking_regions(fasta_path=fasta_path, gff_path=gff_path, output_path=output_path,
                                        extract_one_hot_flag=False, extract_string_flag=True, extragenic=10, intragenic=10)
        with open(output_path, "r") as f:
            self.assertTrue(f.readline().startswith(">"))
            extract = ""
            for curr_line in f:
                if not curr_line.startswith(">"):
                    extract += curr_line
                else:
                    break
            extract = extract.replace("\n", "")
        target_output = "A" + 10 * "C" + 9 * "T" + 20 * "N" + 10 * "G" + 10 * "A"
        self.assertEqual(extract, target_output)

    def test_string_extraction_reverse(self):
        test_data_path = os.path.join(os.path.dirname(__file__), "test_data")
        fasta_path = os.path.join(test_data_path, "test_input_short.fa")
        gff_path = os.path.join(test_data_path, "test_input_short_reverse.gff3")
        output_path = os.path.join(test_data_path, "test_result_short.txt")
        write_mockup_data(repeat_length=10, file_name="test_input_short.fa")
        ih.extract_gene_flanking_regions(fasta_path=fasta_path, gff_path=gff_path, output_path=output_path,
                                        extract_one_hot_flag=False, extract_string_flag=True, extragenic=10, intragenic=10)
        with open(output_path, "r") as f:
            self.assertTrue(f.readline().startswith(">"))
            extract = ""
            for curr_line in f:
                if not curr_line.startswith(">"):
                    extract += curr_line
                else:
                    break
            extract = extract.replace("\n", "")
        target_output = 10 * "T" + 10 * "C" + 20 * "N" + 9 * "A" + 10 * "G" + "T"
        self.assertEqual(extract, target_output)

    def test_array_extraction(self):
        test_data_path = os.path.join(os.path.dirname(__file__), "test_data")
        fasta_path = os.path.join(test_data_path, "test_input_short.fa")
        gff_path = os.path.join(test_data_path, "test_input_short.gff3")
        write_mockup_data(repeat_length=10, file_name="test_input_short.fa")
        result_list = []
        result_list.append(self.encoding["A"])
        result_list.extend([self.encoding["C"] for _ in range(10)])
        result_list.extend([self.encoding["T"] for _ in range(9)])
        result_list.extend([self.encoding["N"] for _ in range(20)])
        result_list.extend([self.encoding["G"] for _ in range(10)])
        result_list.extend([self.encoding["A"] for _ in range(10)])
        result_array = np.array(result_list, dtype=np.float64)
        ih_output = ih.extract_gene_flanking_regions(fasta_path=fasta_path, gff_path=gff_path, extract_one_hot_flag=True,
                                                     extract_string_flag=False, extragenic=10, intragenic=10)
        extracted_seq = ih_output[0][0]
        self.assertTrue(np.array_equal(extracted_seq, result_array))
    
    def test_array_extraction_reverse(self):
        test_data_path = os.path.join(os.path.dirname(__file__), "test_data")
        fasta_path = os.path.join(test_data_path, "test_input_short.fa")
        gff_path = os.path.join(test_data_path, "test_input_short_reverse.gff3")
        write_mockup_data(repeat_length=10, file_name="test_input_short.fa")
        result_list = []
        result_list.extend([self.encoding["T"] for _ in range(10)])
        result_list.extend([self.encoding["C"] for _ in range(10)])
        result_list.extend([self.encoding["N"] for _ in range(20)])
        result_list.extend([self.encoding["A"] for _ in range(9)])
        result_list.extend([self.encoding["G"] for _ in range(10)])
        result_list.append(self.encoding["T"])
        result_array = np.array(result_list, dtype=np.float64)
        ih_output = ih.extract_gene_flanking_regions(fasta_path=fasta_path, gff_path=gff_path, extract_one_hot_flag=True,
                                                     extract_string_flag=False, extragenic=10, intragenic=10)
        extracted_seq = ih_output[0][0]
        # for i, j in zip(result_array, extracted_seq):
        #     print(i, j)
        self.assertTrue(np.array_equal(extracted_seq, result_array))

def write_mockup_data(repeat_length: int = 500, file_name: str = ""):
    sequence = "A" * repeat_length + "C" * repeat_length + "T" * repeat_length + "G" * repeat_length
    sequence = sequence * 5
    name = "1 mock_up_sequence"
    if file_name == "":
        file_name = "test_input.fa"
    fasta_path = os.path.join(os.path.dirname(__file__), "test_data", file_name)
    with open(fasta_path, "w") as f:
        f.write(f">{name}\n")
        for i in range(0, len(sequence) + 1, 80):
            f.write(f"{sequence[i:i+80]}\n")
        if i != len(sequence):
            f.write(f"{sequence[i+80:-1]}\n")
    


if __name__ == "__main__":
    #todo: 
    #   make sure this works with string as well as one hot extraction
    #       off by one error in extraction?
    #   add argument parser to extract stuff from anywhere, if necessary??
    # ih.extract_gene_flanking_regions(fasta_path="/home/gernot/Downloads/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa",
    #                                  gff_path="/home/gernot/Downloads/Arabidopsis_thaliana.TAIR10.59.gff3",
    #                                  output_path="test.fa", extract_one_hot_flag=False, extract_string_flag=True, 
    #                                  extragenic=100, intragenic=50)
    # write_mockup_data(repeat_length=10)
    # print(os.getcwd())
    # ih.extract_gene_flanking_regions(fasta_path="/home/gernot/Code/PhD_Code/DeepCRE_Collab/test_model/test_data/test_input_short.fa",
    #                                  gff_path="/home/gernot/Code/PhD_Code/DeepCRE_Collab/test_model/test_data/test_input_short.gff3",
    #                                  output_path="/home/gernot/Code/PhD_Code/DeepCRE_Collab/test_model/test_data/test_result_short.txt",
    #                                  extract_one_hot_flag=False, extract_string_flag=True, extragenic=10, intragenic=10)
    unittest.main()

    # test_fasta()
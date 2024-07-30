###testmodel.py###

import pandas as pd
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras import models     #type:ignore
from tensorflow.keras import backend    #type:ignore


tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()  # changed from enable 23-july 8:55
#tf.config.set_visible_devices([], 'GPU') # changed from enable 23-july 9:18

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' #added for GPU
import numpy as np
import deeplift
import shap
import shap.explainers.deep.deep_tf   #added on 15:22 22-july
from deeplift.util import get_shuffle_seq_ref_function
from deeplift.dinuc_shuffle import dinuc_shuffle
from deeplift.conversion import kerasapi_conversion as kc
from deeplift.util import get_hypothetical_contribs_func_onehot
from sklearn.metrics import accuracy_score
import h5py
import unittest
import sys
model_path = os.path.join(os.path.dirname(__file__), "..", "model")
sys.path.insert(1, model_path)
from utils import prepare_valid_seqs
from motif_discovery_ssr_leaf import compute_scores, combine_mult_and_diffref

from collections import Counter                                                        #added on 15:59 22-july

#This generates 20 references per sequence
def shuffle_several_times(list_containing_input_modes_for_an_example,                  #"dinuc_" removed on 15:59 22-july
                                seed=1234):                                            # taken from colab link
  assert len(list_containing_input_modes_for_an_example)==1
  onehot_seq = list_containing_input_modes_for_an_example[0]
  rng = np.random.RandomState(seed)
  to_return = np.array([dinuc_shuffle(onehot_seq, rng=rng) for i in range(20)])
  return [to_return] #wrap in list for compatibility with multiple modes


#def shuffle_several_times(seqs,reps:int=100):                                          #added on 15:43 22-july
#    seqs = np.array(seqs)
#    assert len(seqs.shape) == 3
#    sep_shuffled_seqs = np.array([dinuc_shuffle(s, num_shufs=reps) for s in seqs])
#    shuffle_out = rearrange(sep_shuffled_seqs, "b r l n -> (b r) l n")
#    return shuffle_out

class TestShap(unittest.TestCase):
    def compute_actual_and_hypothetical_scores(self, fasta, gtf, tpms, specie):
        for saved_model_name_keras in os.listdir(os.path.join(model_path, 'saved_models')):
            if saved_model_name_keras.startswith(specie) and saved_model_name_keras.endswith('terminator.h5'):
                print(saved_model_name_keras)
                val_chrom = saved_model_name_keras.split('_')[2]
                x_val, y_val, genes_val = prepare_valid_seqs(fasta, gtf, tpms, val_chrom, pkey=False)

                saved_model_path_keras = os.path.join(model_path, 'saved_models', saved_model_name_keras)
                saved_model_path_h5 = os.path.splitext(saved_model_path_keras)[0] + ".h5"
                loaded_model = tf.keras.models.load_model(saved_model_path_keras)
                input_shape = x_val.shape
                # print(loaded_model.summary())
                predicted_prob = loaded_model.predict(x_val)
                predicted_class = predicted_prob > 0.5
                print('Accuracy', accuracy_score(y_val, predicted_class))
                x = []
                for idx, seq in enumerate(x_val):
                    if predicted_class[idx] == 0 and y_val[idx] == 0:
                        x.append(seq)
                for idx, seq in enumerate(x_val):
                    if predicted_class[idx] == 1 and y_val[idx] == 1:
                        x.append(seq)

                x = np.array(x)

                print(f'Number of correct predictions {x.shape[0]}')
                raw_shap_explanations = compute_scores(onehot_data=x, keras_model=loaded_model, hypothetical=False)
                hyp_scores = compute_scores(onehot_data=x, keras_model=loaded_model, hypothetical=True)
                print(f"raw_shap_explanations:\n {raw_shap_explanations}")
                print(f"hyp_shap_explanations:\n {hyp_scores}")
                print(f"input_shape: {x_val.shape}")
                print(f"raw explanations shape: {raw_shap_explanations.shape}")
                print(f"hyp explanations shape: {hyp_scores.shape}")
                self.assertIsNotNone(raw_shap_explanations)
                self.assertEquals(hyp_scores[0].shape, raw_shap_explanations.shape)
                return raw_shap_explanations


    def test_shap_lift(self):
        os.chdir(model_path)
        species = ['zea']
        gene_models = ['Zea_mays.Zm-B73-REFERENCE-NAM-5.0.52.gtf']
        genomes = ['Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.toplevel.fa']
        pickle_keys = ["zea"]
        mapped_read_counts = ['zea_counts.csv']

        for plant, fasta_file, gtf_file, pickled_key, counts in zip(species, genomes, gene_models, pickle_keys,
                                                                    mapped_read_counts):
            if not os.path.exists(f'modisco/{plant}_modisco.hdf5'):
                print(f'Computing contribution and hypothetical contribution scores for {plant}-----------------------------\n')
                res = self.compute_actual_and_hypothetical_scores(fasta_file, gtf_file, counts, plant)
                print("resulting explanations: ", res)
    
    # def test_deep_shap(self):
    #     model = keras.Sequential([
    #         layers.Dense(2, activation="relu", name="layer1"),
    #         layers.Dense(3, activation="relu", name="layer2"),
    #         layers.Dense(4, name="layer3"),
    #     ])
    #     x = tf.ones((3, 3))
    #     y = model(x)
    #     shap_model = shap.DeepExplainer((x, y), shuffle_several_times)
    #     raw_shap_explanations = shap_model.shap_values(x, check_additivity=False)
    #     print(raw_shap_explanations)
    #     self.assertTrue(raw_shap_explanations)




if __name__ == "__main__":
    # h5_path = "/home/gernot/Code/PhD_Code/DeepCRE_Collab/model/saved_models/arabidopsis_model_1_promoter_terminator.keras"
    # model = tf.keras.models.load_model(h5_path)
    unittest.main()

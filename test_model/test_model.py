
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras import models     #type:ignore
from tensorflow.keras import backend    #type:ignore

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()
tf.config.set_visible_devices([], 'GPU')

import os
import numpy as np
import deeplift
import shap
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
# from motif_discovery_ssr_leaf import shuffle_several_times

class TestShap(unittest.TestCase):


    @staticmethod
    def compute_shap_scores(seqs, model, tf_idx=0):
        shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough#type:ignore
        shap.explainers.deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers.deep.deep_tf.linearity_1d(0)#type:ignore
        dinuc_shuff_explainer = shap.DeepExplainer(
            (model.input, model.output[:, tf_idx]),
            data=shuffle_several_times,
            combine_mult_and_diffref=combine_mult_and_diffref) #type:ignore
        hypothetical_scores = dinuc_shuff_explainer.shap_values(seqs)
        actual_scores = hypothetical_scores * seqs#type:ignore
        return actual_scores
    
    @staticmethod
    def compute_scores(onehot_data, keras_model):
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
        shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough
        dinuc_shuff_explainer = shap.DeepExplainer(model=(keras_model.input, keras_model.output[:, 0]),
                                                data=shuffle_several_times)
        raw_shap_explanations = dinuc_shuff_explainer.shap_values(onehot_data, check_additivity=False)
        dinuc_shuff_explanations = (np.sum(raw_shap_explanations, axis=-1)[:, :, None] * onehot_data)
        print("im here!")
        print(dinuc_shuff_explanations)

        return dinuc_shuff_explanations

    def compute_actual_and_hypothetical_scores(self, fasta, gtf, tpms, specie):
        for saved_model_name_keras in os.listdir(os.path.join(model_path, 'saved_models')):
            if saved_model_name_keras.startswith(specie) and saved_model_name_keras.endswith('terminator.keras'):
                print(saved_model_name_keras)
                val_chrom = saved_model_name_keras.split('_')[2]
                x_val, y_val, genes_val = prepare_valid_seqs(fasta, gtf, tpms, val_chrom, pkey=False)

                saved_model_path_keras = os.path.join(model_path, 'saved_models', saved_model_name_keras)
                saved_model_path_h5 = os.path.splitext(saved_model_path_keras)[0] + ".h5"
                loaded_model = tf.keras.models.load_model(saved_model_path_keras)
                input_shape = x_val.shape
                print(input_shape)
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
                print(x_val)

                print(f'Number of correct predictions {x.shape[0]}')
                raw_shap_explanations = self.compute_scores(onehot_data=x, keras_model=loaded_model)
                # ---------- Computing importance and hypothetical scores-------------------------------------------#
                # deeplift_model = kc.convert_model_from_saved_files("/home/gernot/Code/PhD_Code/DeepCRE_Collab/model/saved_models/arabidopsis_model_1_promoter_terminator.h5",
                #                                                 nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault) #type:ignore
                # shap_model = shap.DeepExplainer((x_val, predicted_prob[:, 0]), shuffle_several_times)

                # deeplift_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=0,
                #                                                                 target_layer_idx=-2)

                # contribs_many_refs_func = get_shuffle_seq_ref_function(
                #     score_computation_function=deeplift_contribs_func,
                #     shuffle_func=dinuc_shuffle)

                # multipliers_func = deeplift_model.get_target_multipliers_func(find_scores_layer_idx=0,
                #                                                             target_layer_idx=-2)
                # hypothetical_contribs_func = get_hypothetical_contribs_func_onehot(multipliers_func)

                # # Once again, we rely on multiple shuffled references
                # hypothetical_contribs_many_refs_func = get_shuffle_seq_ref_function(
                #     score_computation_function=hypothetical_contribs_func,
                #     shuffle_func=dinuc_shuffle)

                # actual_scores = np.squeeze(np.sum(contribs_many_refs_func(task_idx=0,
                #                                                         input_data_sequences=x,
                #                                                         num_refs_per_seq=10,
                #                                                         batch_size=50,
                #                                                         progress_update=4000), axis=2))[:, :, None] * x

                # raw_shap_explanations = shap_model.shap_values(x, check_additivity=False)
                print(raw_shap_explanations)
                print(raw_shap_explanations.shape)
                # print(actual_scores)
                # print(actual_scores.shape)
                self.assertIsNotNone(raw_shap_explanations)
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
                print(f'Running TFMoDisco on {plant}------------------------------------------------------------------------\n')
    
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

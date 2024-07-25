import pandas as pd
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras import models     #type:ignore
from tensorflow.keras import backend    #type:ignore

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()  # changed from enable 23-july 8:55

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
import modisco
from importlib import reload
from utils import prepare_valid_seqs
#from einops import rearrange


model_path = os.path.join(os.path.dirname(__file__), "..", "model")


#def shuffle_several_times_fritz(seqs, reps: int = 100):
#    """creates shuffled versions of input data
#
#    Args:
#        s (_type_): input data to be shuffled
#        num_shufs (int, optional): number of shuffled versions of the input to be created. Defaults to 100.
#
#    Returns:
#        _type_: returns multiple shuffled versions of the input
#    """
#    seqs = np.array(seqs)
#    assert len(seqs.shape) == 3
#    sep_shuffled_seqs = np.array([dinuc_shuffle(s, num_shufs=reps) for s in seqs])
#    shuffle_out = rearrange(sep_shuffled_seqs, "b r l n -> (b r) l n")
#    return shuffle_out


#This generates 20 references per sequence
def shuffle_several_times(list_containing_input_modes_for_an_example,                  #"dinuc_" removed on 15:59 22-july
                                seed=1234):                                            # taken from colab link
  assert len(list_containing_input_modes_for_an_example)==1
  onehot_seq = list_containing_input_modes_for_an_example[0]
  rng = np.random.RandomState(seed)
  to_return = np.array([dinuc_shuffle(onehot_seq, rng=rng) for i in range(20)])
  return [to_return] #wrap in list for compatibility with multiple modes


def combine_mult_and_diffref(mult, orig_inp, bg_data):
    to_return = []
    for l in range(len(mult)):
        projected_hypothetical_contribs = np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape)==2
        #At each position in the input sequence, we iterate over the one-hot encoding
        # possibilities (eg: for genomic sequence, this is ACGT i.e.
        # 1000, 0100, 0010 and 0001) and compute the hypothetical 
        # difference-from-reference in each case. We then multiply the hypothetical
        # differences-from-reference with the multipliers to get the hypothetical contributions.
        #For each of the one-hot encoding possibilities,
        # the hypothetical contributions are then summed across the ACGT axis to estimate
        # the total hypothetical contribution of each position. This per-position hypothetical
        # contribution is then assigned ("projected") onto whichever base was present in the
        # hypothetical sequence.
        #The reason this is a fast estimate of what the importance scores *would* look
        # like if different bases were present in the underlying sequence is that
        # the multipliers are computed once using the original sequence, and are not
        # computed again for each hypothetical sequence.
        for i in range(orig_inp[l].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:,i] = 1.0
            hypothetical_difference_from_reference = (hypothetical_input[None,:,:]-bg_data[l])
            hypothetical_contribs = hypothetical_difference_from_reference*mult[l]
            projected_hypothetical_contribs[:,:,i] = np.sum(hypothetical_contribs,axis=-1) 
        to_return.append(np.mean(projected_hypothetical_contribs,axis=0))
    return to_return


def compute_scores(onehot_data, keras_model, hypothetical=False):
    shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough
    shap.explainers.deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers.deep.deep_tf.passthrough  #removed "_" from _deep on 15:22 22-july
    if hypothetical:
        dinuc_shuff_explainer = shap.DeepExplainer(model=(keras_model.input, keras_model.output[:, 0]),
                                                   data=shuffle_several_times, combine_mult_and_diffref=combine_mult_and_diffref)
    else:
        dinuc_shuff_explainer = shap.DeepExplainer(model=(keras_model.input, keras_model.output[:, 0]),
                                                   data=shuffle_several_times)
    raw_shap_explanations = dinuc_shuff_explainer.shap_values(onehot_data)  #23-july 8:57 removed ", check_additivity=False"
    dinuc_shuff_explanations = (np.sum(raw_shap_explanations, axis=-1)[:, :, None] * onehot_data)
    print(dinuc_shuff_explanations)
    return dinuc_shuff_explanations


def compute_actual_and_hypothetical_scores(fasta, gtf, tpms, specie, save_files=True):
    actual_scores_all, hypothetical_scores_all, onehot_all = [], [], []
    for saved_model_file_name in os.listdir(os.path.join(model_path, 'saved_models')):
        if saved_model_file_name.startswith(specie) and saved_model_file_name.endswith('terminator.h5'):
            print(saved_model_file_name)
            val_chrom = saved_model_file_name.split('_')[2]
            x_val, y_val, genes_val = prepare_valid_seqs(fasta, gtf, tpms, val_chrom, pkey=False)

            saved_model_path = os.path.join(os.path.dirname(__file__), 'saved_models', saved_model_file_name)
            loaded_model = tf.keras.models.load_model(saved_model_path)
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
            # ---------- Computing importance and hypothetical scores-------------------------------------------#
            # deeplift_model = kc.convert_model_from_saved_files(saved_model_path,
            #                                                    nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault) #type:ignore
            # # deeplift_model = shap.DeepExplainer((loaded_model.input, loaded_model.output[:, 0]), shuffle_several_times)

            # deeplift_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=0,
            #                                                                  target_layer_idx=-2)

            # contribs_many_refs_func = get_shuffle_seq_ref_function(
            #     score_computation_function=deeplift_contribs_func,
            #     shuffle_func=dinuc_shuffle)

            # multipliers_func = deeplift_model.get_target_multipliers_func(find_scores_layer_idx=0,
            #                                                               target_layer_idx=-2)
            # hypothetical_contribs_func = get_hypothetical_contribs_func_onehot(multipliers_func)

            # # Once again, we rely on multiple shuffled references
            # hypothetical_contribs_many_refs_func = get_shuffle_seq_ref_function(
            #     score_computation_function=hypothetical_contribs_func,
            #     shuffle_func=dinuc_shuffle)

            actual_scores = compute_scores(onehot_data=x, keras_model=loaded_model, hypothetical=False)
            # actual_scores = np.squeeze(np.sum(contribs_many_refs_func(task_idx=0,
            #                                                           input_data_sequences=x,
            #                                                           num_refs_per_seq=10,
            #                                                           batch_size=50,
            #                                                           progress_update=4000), axis=2))[:, :, None] * x

            hyp_scores = compute_scores(onehot_data=x, keras_model=loaded_model, hypothetical=True)
            # hyp_scores = hypothetical_contribs_many_refs_func(task_idx=0,
            #                                                   input_data_sequences=x,
            #                                                   num_refs_per_seq=10,
            #                                                   batch_size=50,
            #                                                   progress_update=4000)

            print("ACTUAL SCORES:")
            print(actual_scores)
            print("HYPOTHETICAL SCORES:")
            print(hyp_scores)
            actual_scores_all.append(actual_scores)
            hypothetical_scores_all.append(hyp_scores)
            onehot_all.append(x)

    if len(actual_scores_all) > 1 and save_files:
        # Save scores in h5 format
        if os.path.isfile(f'modisco/{specie}_scores.h5'):
            os.system(f'rm -rf modisco/{specie}_scores.h5')

        actual_scores_all = np.concatenate(actual_scores_all, axis=0)
        hypothetical_scores_all = np.concatenate(hypothetical_scores_all, axis=0)
        onehot_all = np.concatenate(onehot_all, axis=0)

        h = h5py.File(f'modisco/{specie}_scores.h5', 'w')
        h.create_dataset('contrib_scores', data=actual_scores_all)
        h.create_dataset('hypothetical_scores', data=hypothetical_scores_all)
        h.create_dataset('one_hots', data=onehot_all)
        h.close()
    elif not save_files:
        print("not saving anything since this is a test run!")
    else:
        print(f"specie {specie} showed no results to save!")


def run_modisco(specie):
    species_score_path = f'modisco/{specie}_scores.h5'
    save_file = f"modisco/{specie}_modisco.hdf5"
    if not os.path.isfile(species_score_path):
        print(f"no score files found for {specie}! Aborting modisco!")
        return 

    os.system(f'rm -rf {save_file}')

    h5_data = h5py.File(species_score_path, 'r')
    contribution_scores = h5_data.get('contrib_scores')
    hypothetical_scores = h5_data.get('hypothetical_scores')
    one_hots = h5_data.get('one_hots')

    print('contributions', contribution_scores.shape)               #type:ignore
    print('hypothetical contributions', hypothetical_scores.shape)  #type:ignore
    print('correct predictions', one_hots.shape)                    #type:ignore
    # -----------------------Running modisco----------------------------------------------#
    # Uncomment to refresh modules for when tweaking code during development:
    reload(modisco.util)
    reload(modisco.pattern_filterer)
    reload(modisco.aggregator)
    reload(modisco.core)
    reload(modisco.seqlet_embedding.advanced_gapped_kmer)
    reload(modisco.affinitymat.transformers)
    reload(modisco.affinitymat.core)
    reload(modisco.affinitymat)
    reload(modisco.cluster.core)
    reload(modisco.cluster)
    reload(modisco.tfmodisco_workflow.seqlets_to_patterns)
    reload(modisco.tfmodisco_workflow)
    reload(modisco)

    null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=5000)
    tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
        # Slight modifications from the default settings
        sliding_window_size=21,
        flank_size=5,
        target_seqlet_fdr=0.01,
        seqlets_to_patterns_factory=modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
            # Note: as of version 0.5.6.0, it's possible to use the results of a motif discovery
            # software like MEME to improve the TF-MoDISco clustering. To use the meme-based
            # initialization, you would specify the initclusterer_factory as shown in the
            # commented-out code below:
            # initclusterer_factory=modisco.clusterinit.memeinit.MemeInitClustererFactory(
            #    meme_command="meme", base_outdir="meme_out",
            #    max_num_seqlets_to_use=10000, nmotifs=10, n_jobs=1),
            trim_to_window_size=10,
            initial_flank_to_add=2,
            final_flank_to_add=0,
            final_min_cluster_size=60,
            # use_pynnd=True can be used for faster nn comp at coarse grained step
            # (it will use pynndescent), but note that pynndescent may crash
            # use_pynnd=True,
            n_cores=50)
    )(
        task_names=['task0'],
        contrib_scores={'task0': contribution_scores},
        hypothetical_contribs={'task0': hypothetical_scores},
        one_hot=one_hots,
        null_per_pos_scores=null_per_pos_scores)

    reload(modisco.util)
    grp = h5py.File(save_file, "w")
    tfmodisco_results.save_hdf5(grp)
    grp.close()


def main(test=False):
    backend.clear_session()
    if not os.path.exists('modisco'):
        os.mkdir('modisco')
    os.chdir(model_path)
    species = ['arabidopsis', 'zea', 'solanum', 'sbicolor']
    gene_models = ['Arabidopsis_thaliana.TAIR10.52.gtf', 'Zea_mays.Zm-B73-REFERENCE-NAM-5.0.52.gtf',
                'Solanum_lycopersicum.SL3.0.52.gtf', 'Sorghum_bicolor.Sorghum_bicolor_NCBIv3.52.gtf']
    genomes = ['Arabidopsis_thaliana.TAIR10.dna.toplevel.fa', 'Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.toplevel.fa',
            'Solanum_lycopersicum.SL3.0.dna.toplevel.fa', 'Sorghum_bicolor.Sorghum_bicolor_NCBIv3.dna.toplevel.fa']
    pickle_keys = ['ara', 'zea', 'sol', 'sor']
    mapped_read_counts = ['arabidopsis_counts.csv', 'zea_counts.csv', 'solanum_counts.csv', 'sbicolor_counts.csv']

    for plant, fasta_file, gtf_file, pickled_key, counts in zip(species, genomes, gene_models, pickle_keys,
                                                                mapped_read_counts):
        if not os.path.exists(f'modisco/{plant}_modisco.hdf5'):
            print(f'Computing contribution and hypothetical contribution scores for {plant}-----------------------------\n')
            compute_actual_and_hypothetical_scores(fasta_file, gtf_file, counts, plant, save_files=test)
            if not test:
                print(f'Running TFMoDisco on {plant}------------------------------------------------------------------------\n')
                run_modisco(plant)


# def load_tf1(path, input):
#   print('Loading from', path)
#   with tf.Graph().as_default() as g:
#     with tf.compat.v1.Session() as sess:
#       meta_graph = tf.compat.v1.saved_model.load(sess, ["serve"], path)
#       sig_def = meta_graph.signature_def[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
#       input_name = sig_def.inputs['input'].name
#       output_name = sig_def.outputs['output'].name
#       print('  Output with input', input, ': ', 
#             sess.run(output_name, feed_dict={input_name: input}))



if __name__ == "__main__":
    # h5_path = "/home/gernot/Code/PhD_Code/DeepCRE_Collab/model/saved_models/arabidopsis_model_1_promoter_terminator.keras"
    # model = tf.keras.models.load_model(h5_path)
    main(test=True)

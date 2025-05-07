from openfold_func import *
import argparse
import os
from tqdm import tqdm
import time
import sys
import subprocess
import torch


sys.path.append("/home/gluetown/butterfly/openfold/")
sys.path.append("/home/gluetown/butterfly/ColabFold/")
from colabfold.colabfold import run_mmseqs2

from colabfold.utils import (
    ACCEPT_DEFAULT_TERMS,
    DEFAULT_API_SERVER,
    NO_GPU_FOUND,
    CIF_REVISION_DATE,
    get_commit,
    safe_filename,
    setup_logging,
    CFMMCIFIO,
)
from scripts.utils import add_data_args
from openfold.config import model_config
from openfold.model.model import AlphaFold
from run_pretrained_openfold import (
    precompute_alignments,
    list_files_with_extensions,
    generate_feature_dict,
)
from openfold.utils.script_utils import (
    load_models_from_command_line,
    parse_fasta,
    run_model,
    prep_output,
    relax_protein,
)
from openfold.utils.tensor_utils import tensor_tree_map

from openfold.data import templates, feature_pipeline, data_pipeline

from Bio.Seq import Seq
from Bio import SeqIO

if __name__ == "__main__":
    ###########################
    ### INPUT - OUTPUT DIRS ###
    ###########################

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory containing the data")
    parser.add_argument("-m", "--mmCIF_dir", required=True, help="Directory containing mmCIF files")
    parser.add_argument("-a", "--alignment", type=bool, required=True, help="Whether to compute alignments with ColabFold")
    add_data_args(parser)    
    args = parser.parse_args()

    args.input_dir = "/home/gluetown/openfold_embeddings/test/"
    args.mmCIF_dir = "/home/gluetown/butterfly/openfold/data/mmCIF/pdb_mmcif/mmcif_files/"
    args.alignment = True

    input_dir = args.input_dir
    fasta_dir = input_dir + "fasta/"
    alignment_dir = input_dir + "alignment/"
    output_dir = input_dir + "step7_embeddings/"
    compute_alignments = args.alignment
    os.makedirs(alignment_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    ######################
    ### ADJUST OPTIONS ###
    ######################

    ### mmseqs settings
    use_env = "env"
    host_url: str = DEFAULT_API_SERVER
    user_agent: str = "user" # set your user name here

    ### Openfold settings
    
    args.output_dir = output_dir
    args.use_precomputed_alignments = False
    args.config_preset = "model_1_ptm"
    args.model_device  = "cuda:0"
    args.data_random_seed = 42 
    args.use_single_seq_mode = False
    args.template_mmcif_dir = args.mmCIF_dir
    is_multimer = False


    ### Model Configuration ###
    config = model_config(
            args.config_preset, 
            train=False,
            long_sequence_inference=False,
            use_deepspeed_evoformer_attention=False,
            )
    model = AlphaFold(config)
    model.globals.use_deepspeed_evo_attention = True
    # bug with normal attention function where attn_softmax_inplace_forward
    # can't be executed (model + inputs are on gpu, but code attempts to 
    # use cpu)
    model = model.to(args.model_device)
    model.eval()
    # even though model training mode is off, still can't pass one of the 
    # assertions for model training when doing inference

    all_tag_list = []
    all_seq_list = []
    for fasta_file in list_files_with_extensions(fasta_dir, (".fasta", ".fa")):
        # Gather input sequences
        fasta_path = os.path.join(fasta_dir, fasta_file)
        with open(fasta_path, "r") as fp:
            data = fp.read()
        tags, seqs = parse_fasta(data)

        # assert len(tags) == len(set(tags)), "All FASTA tags must be unique"
        tag = '-'.join(tags)

        all_tag_list.append((tag, tags))
        all_seq_list.append(seqs)

    # for some reason, can only process one seq at a time

    ##################
    ### Embeddings ###
    ##################
    for fasta_file in os.listdir(fasta_dir):
        with open(os.path.join(fasta_dir, fasta_file), "r") as file:
            for record in SeqIO.parse(file, "fasta"):
                protein_id = record.id
                protein_dir = os.path.join(alignment_dir, protein_id)
                os.makedirs(protein_dir, exist_ok=True)

    for protein in os.listdir(alignment_dir):
        ### Process fasta files
        print(protein)
        prot_index = all_tag_list[0][1].index(protein)
        tag_list = [(protein), [protein]]
        seq_list = [[all_seq_list[0][prot_index]]]
        tags = [protein]
        seqs = [all_seq_list[0][prot_index]]

        ### CREATE ALIGNMENTS USING COLABFOLD MMSEQS ###
        if compute_alignments:
            seq_dict = {}
            for file in os.listdir(fasta_dir):
                with open(f"{fasta_dir}/{file}", "r") as fasta_file:
                    tmp = {i.id:i.seq for i in list(SeqIO.parse(fasta_file, "fasta"))}
                    seq_dict.update(tmp)
                
            time1 = time.time()
            for seq_id, seq  in tqdm(seq_dict.items()):
                seq = str(seq)
                seq_dir = f"{alignment_dir}/{seq_id}/"
                if not os.path.exists(seq_dir):
                    os.makedirs(seq_dir, exist_ok=True)
                    print(f"Processing {seq_id}\nWriting to {seq_dir}")
                    run_mmseqs2(    
                                seq,
                                seq_dir,
                                use_env,
                                use_pairing=False,
                                host_url=host_url,
                                user_agent=user_agent,
                                )
            time2 = time.time()
            print(f"Time taken: {time2-time1}")

            ### Clean Alignments ### 
            # for some reason, there's a \0 or \x00 chr (null) added to the 
            # end of all the aln files (.a3m)
            clean_script = "clean_aln.sh"
            
            try:
                subprocess.run(["bash", clean_script, alignment_dir], check=True)
                print(f"Successfully cleaned alignments in {input_dir}")
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while running {clean_script}: {e}")

        ########################################################
        ### CREATE FEATURE DICT AS INPUT TO ALPHAFOLD2 MODEL ###
        ########################################################

        ### Data Preprocessing ###
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=config.data.predict.max_templates,
            kalign_binary_path=args.kalign_binary_path,
            release_dates_path=args.release_dates_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path
        )
        data_processor = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer,
            )
        feature_dict = generate_feature_dict(
                            tags,
                            seqs,
                            alignment_dir,
                            data_processor,
                            args,
                        )
        feature_processor = feature_pipeline.FeaturePipeline(config.data)
        processed_feature_dict = feature_processor.process_features(
                        feature_dict, mode='predict', is_multimer=is_multimer
                    )
        processed_feature_dict = {
                        k: torch.as_tensor(v, device=args.model_device)
                        for k, v in processed_feature_dict.items()
                    }

        ######################################################
        ### GET EMBEDDINGS FROM OUTPUT OF STRUCTURE MODULE ###
        ######################################################
        batch = processed_feature_dict
        cycle_no = 0
        fetch_cur_batch = lambda t: t[..., cycle_no]
        feats = tensor_tree_map(fetch_cur_batch, batch)
        m_1_prev, z_prev, x_prev = None, None, None
        prevs = [m_1_prev, z_prev, x_prev]
        num_iters = batch["aatype"].shape[-1]

        ### Get output of evoformer ###
        evoformer_output_dict = get_evoformer_outputs(model,
                                        feats,
                                        prevs,
                                        _recycle=(num_iters > 1)
                                    )

        ### Get embeddings from Structure Module ###
        # Cut before Step 7--> to change, modify get_structure_module_embeddings function
        structure_module = model.structure_module
        embeddings = get_structure_module_embeddings(structure_module, 
                                        evoformer_output_dict,
                                        feats["aatype"],
                                        mask=None,
                                        inplace_safe=False,
                                        _offload_inference=False,
                                        )     
        print(f"Saving {protein} embeddings to {output_dir}/{protein}.pt")
        torch.save(embeddings, f"{output_dir}/{protein}.pt")

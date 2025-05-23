import os
import sys
import argparse
import time
import subprocess
from tqdm import tqdm

import torch
from Bio.Seq import Seq
from Bio import SeqIO

# Add OpenFold to the Python path
sys.path.append("your_dir/openfold/")
sys.path.append("your_dir/ColabFold/")

# Import OpenFold modules
from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.data.tools import hhsearch, hmmsearch
from openfold.np import protein
from openfold.utils.script_utils import (
    load_models_from_command_line,
    parse_fasta,
    run_model,
    prep_output,
    relax_protein,
)
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.trace_utils import (
    pad_feature_dict_seq,
    trace_model_,
)
from openfold.model.model import AlphaFold
from openfold.utils.feats import (
    pseudo_beta_fn,
    build_extra_msa_feat,
    build_template_angle_feat,
    build_template_pair_feat,
    atom14_to_atom37,
)
from openfold.utils.tensor_utils import add, dict_multimap
import openfold.np.residue_constants as residue_constants
from openfold.utils.rigid_utils import Rotation, Rigid

# Import ColabFold modules
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

# Import precompute and utility scripts
from run_pretrained_openfold import (
    precompute_alignments,
    list_files_with_extensions,
    generate_feature_dict,
)
from scripts.precompute_embeddings import EmbeddingGenerator
from scripts.utils import add_data_args

# Torch configuration
torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if (
    torch_major_version > 1 or
    (torch_major_version == 1 and torch_minor_version >= 12)
):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

torch.set_grad_enabled(False)


### Data Preprocessing ###
def precompute_alignments(tags, seqs, alignment_dir, args):
    for tag, seq in zip(tags, seqs):
        tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(alignment_dir, tag)

        os.makedirs(local_alignment_dir, exist_ok=True)

        template_searcher = hhsearch.HHSearch(
            binary_path=args.hhsearch_binary_path,
            databases=[args.pdb70_database_path],
        )

        alignment_runner = data_pipeline.AlignmentRunner(
            jackhmmer_binary_path=args.jackhmmer_binary_path,
            hhblits_binary_path=args.hhblits_binary_path,
            uniref90_database_path=args.uniref90_database_path,
            mgnify_database_path=args.mgnify_database_path,
            bfd_database_path=args.bfd_database_path,
            uniref30_database_path=args.uniref30_database_path,
            uniclust30_database_path=args.uniclust30_database_path,
            uniprot_database_path=args.uniprot_database_path,
            template_searcher=template_searcher,
            use_small_bfd=args.bfd_database_path is None,
            no_cpus=args.cpus
        )

        alignment_runner.run(
            tmp_fasta_path, local_alignment_dir
        )
    
        # Remove temporary FASTA file
        os.remove(tmp_fasta_path)

def generate_feature_dict(
    tags,
    seqs,
    alignment_dir,
    data_processor,
    args,
):
    tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")

    if "multimer" in args.config_preset:
        with open(tmp_fasta_path, "w") as fp:
            fp.write(
                '\n'.join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)])
            )
        feature_dict = data_processor.process_fasta(
            fasta_path=tmp_fasta_path, alignment_dir=alignment_dir,
        )
    elif len(seqs) == 1:
        tag = tags[0]
        seq = seqs[0]
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(alignment_dir, tag)
        feature_dict = data_processor.process_fasta(
            fasta_path=tmp_fasta_path,
            alignment_dir=local_alignment_dir,
            seqemb_mode=args.use_single_seq_mode,
        )
    else:
        with open(tmp_fasta_path, "w") as fp:
            fp.write(
                '\n'.join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)])
            )
        feature_dict = data_processor.process_multiseq_fasta(
            fasta_path=tmp_fasta_path, super_alignment_dir=alignment_dir,
        )

    # Remove temporary FASTA file
    os.remove(tmp_fasta_path)

    return feature_dict

#### Inference ###
def iteration(model, feats, prevs, _recycle=True):
    # Primary output dictionary
    outputs = {}

    # This needs to be done manually for DeepSpeed's sake
    dtype = next(model.parameters()).dtype
    for k in feats:
        if feats[k].dtype == torch.float32:
            feats[k] = feats[k].to(dtype=dtype)

    # Grab some data about the input
    batch_dims = feats["target_feat"].shape[:-2]
    no_batch_dims = len(batch_dims)
    n = feats["target_feat"].shape[-2]
    n_seq = feats["msa_feat"].shape[-3]
    device = feats["target_feat"].device

    # Controls whether the model uses in-place operations throughout
    # The dual condition accounts for activation checkpoints
    inplace_safe = False

    # Prep some features
    seq_mask = feats["seq_mask"]
    pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
    msa_mask = feats["msa_mask"]

    if model.globals.is_multimer:
        # Initialize the MSA and pair representations
        # m: [*, S_c, N, C_m]
        # z: [*, N, N, C_z]
        m, z = model.input_embedder(feats)
    elif model.seqemb_mode:
        # Initialize the SingleSeq and pair representations
        # m: [*, 1, N, C_m]
        # z: [*, N, N, C_z]
        m, z = model.input_embedder(
            feats["target_feat"],
            feats["residue_index"],
            feats["seq_embedding"]
        )
    else:
        # Initialize the MSA and pair representations
        # m: [*, S_c, N, C_m]
        # z: [*, N, N, C_z]
        m, z = model.input_embedder(
            feats["target_feat"],
            feats["residue_index"],
            feats["msa_feat"],
            inplace_safe=inplace_safe,
        )

    # Unpack the recycling embeddings. Removing them from the list allows 
    # them to be freed further down in this function, saving memory
    m_1_prev, z_prev, x_prev = reversed([prevs.pop() for _ in range(3)]) 

    # Initialize the recycling embeddings, if needs be 
    if None in [m_1_prev, z_prev, x_prev]:
        # [*, N, C_m]
        m_1_prev = m.new_zeros(
            (*batch_dims, n, model.config.input_embedder.c_m),
            requires_grad=False,
        )

        # [*, N, N, C_z]
        z_prev = z.new_zeros(
            (*batch_dims, n, n, model.config.input_embedder.c_z),
            requires_grad=False,
        )

        # [*, N, 3]
        x_prev = z.new_zeros(
            (*batch_dims, n, residue_constants.atom_type_num, 3),
            requires_grad=False,
        )

    pseudo_beta_x_prev = pseudo_beta_fn(
        feats["aatype"], x_prev, None
    ).to(dtype=z.dtype)

    # The recycling embedder is memory-intensive, so we offload first
    if model.globals.offload_inference and inplace_safe:
        m = m.cpu()
        z = z.cpu()

    # m_1_prev_emb: [*, N, C_m]
    # z_prev_emb: [*, N, N, C_z]
    m_1_prev_emb, z_prev_emb = model.recycling_embedder(
        m_1_prev,
        z_prev,
        pseudo_beta_x_prev,
        inplace_safe=inplace_safe,
    )

    del pseudo_beta_x_prev

    if model.globals.offload_inference and inplace_safe:
        m = m.to(m_1_prev_emb.device)
        z = z.to(z_prev.device)

    # [*, S_c, N, C_m]
    m[..., 0, :, :] += m_1_prev_emb

    # [*, N, N, C_z]
    z = add(z, z_prev_emb, inplace=inplace_safe)

    # Deletions like these become significant for inference with large N,
    # where they free unused tensors and remove references to others such
    # that they can be offloaded later
    del m_1_prev, z_prev, m_1_prev_emb, z_prev_emb

    # Embed the templates + merge with MSA/pair embeddings
    if model.config.template.enabled:
        template_feats = {
            k: v for k, v in feats.items() if k.startswith("template_")
        }

        template_embeds = model.embed_templates(
            template_feats,
            feats,
            z,
            pair_mask.to(dtype=z.dtype),
            no_batch_dims,
            inplace_safe=inplace_safe,
        )

        # [*, N, N, C_z]
        z = add(z,
                template_embeds.pop("template_pair_embedding"),
                inplace_safe,
                )

        if (
            "template_single_embedding" in template_embeds
        ):
            # [*, S = S_c + S_t, N, C_m]
            m = torch.cat(
                [m, template_embeds["template_single_embedding"]],
                dim=-3
            )

            # [*, S, N]
            if not model.globals.is_multimer:
                torsion_angles_mask = feats["template_torsion_angles_mask"]
                msa_mask = torch.cat(
                    [feats["msa_mask"], torsion_angles_mask[..., 2]],
                    dim=-2
                )
            else:
                msa_mask = torch.cat(
                    [feats["msa_mask"], template_embeds["template_mask"]],
                    dim=-2,
                )

    # Embed extra MSA features + merge with pairwise embeddings
    if model.config.extra_msa.enabled:
        if model.globals.is_multimer:
            extra_msa_fn = data_transforms_multimer.build_extra_msa_feat
        else:
            extra_msa_fn = build_extra_msa_feat

        # [*, S_e, N, C_e]
        extra_msa_feat = extra_msa_fn(feats).to(dtype=z.dtype)
        a = model.extra_msa_embedder(extra_msa_feat)

        if model.globals.offload_inference:
            # To allow the extra MSA stack (and later the evoformer) to
            # offload its inputs, we remove all references to them here
            input_tensors = [a, z]
            del a, z

            # [*, N, N, C_z]
            z = model.extra_msa_stack._forward_offload(
                input_tensors,
                msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                chunk_size=model.globals.chunk_size,
                use_deepspeed_evo_attention=model.globals.use_deepspeed_evo_attention,
                use_lma=model.globals.use_lma,
                pair_mask=pair_mask.to(dtype=m.dtype),
                _mask_trans=model.config._mask_trans,
            )

            del input_tensors
        else:
            # [*, N, N, C_z]
            z = model.extra_msa_stack(
                a, z,
                msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                chunk_size=model.globals.chunk_size,
                use_deepspeed_evo_attention=model.globals.use_deepspeed_evo_attention,
                use_lma=model.globals.use_lma,
                pair_mask=pair_mask.to(dtype=m.dtype),
                inplace_safe=inplace_safe,
                _mask_trans=model.config._mask_trans,
            )

    # Run MSA + pair embeddings through the trunk of the network
    # m: [*, S, N, C_m]
    # z: [*, N, N, C_z]
    # s: [*, N, C_s]          
    if model.globals.offload_inference:
        input_tensors = [m, z]
        del m, z
        m, z, s = model.evoformer._forward_offload(
            input_tensors,
            msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
            pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
            chunk_size=model.globals.chunk_size,
            use_deepspeed_evo_attention=model.globals.use_deepspeed_evo_attention,
            use_lma=model.globals.use_lma,
            _mask_trans=model.config._mask_trans,
        )

        del input_tensors
    else:
        m, z, s = model.evoformer(
            m,
            z,
            msa_mask=msa_mask.to(dtype=m.dtype),
            pair_mask=pair_mask.to(dtype=z.dtype),
            chunk_size=model.globals.chunk_size,
            use_deepspeed_evo_attention=model.globals.use_deepspeed_evo_attention,
            use_lma=model.globals.use_lma,
            use_flash=model.globals.use_flash,
            inplace_safe=inplace_safe,
            _mask_trans=model.config._mask_trans,
        )

    outputs["msa"] = m[..., :n_seq, :, :]
    outputs["pair"] = z
    outputs["single"] = s

    del z

    # Predict 3D structure
    outputs["sm"] = model.structure_module(
        outputs,
        feats["aatype"],
        mask=feats["seq_mask"].to(dtype=s.dtype),
        inplace_safe=inplace_safe,
        _offload_inference=model.globals.offload_inference,
    )
    outputs["final_atom_positions"] = atom14_to_atom37(
        outputs["sm"]["positions"][-1], feats
    )
    outputs["final_atom_mask"] = feats["atom37_atom_exists"]
    outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

    # Save embeddings for use during the next recycling iteration

    # [*, N, C_m]
    m_1_prev = m[..., 0, :, :]

    # [*, N, N, C_z]
    z_prev = outputs["pair"]

    early_stop = False
    if model.globals.is_multimer:
        early_stop = model.tolerance_reached(x_prev, outputs["final_atom_positions"], seq_mask)

    del x_prev

    # [*, N, 3]
    x_prev = outputs["final_atom_positions"]

    return outputs, m_1_prev, z_prev, x_prev, early_stop

#### Embeddings ###
def get_evoformer_outputs(model, feats, prevs, _recycle=True):
    # Primary output dictionary
    outputs = {}

    # This needs to be done manually for DeepSpeed's sake
    dtype = next(model.parameters()).dtype
    for k in feats:
        if feats[k].dtype == torch.float32:
            feats[k] = feats[k].to(dtype=dtype)

    # Grab some data about the input
    batch_dims = feats["target_feat"].shape[:-2]
    no_batch_dims = len(batch_dims)
    n = feats["target_feat"].shape[-2]
    n_seq = feats["msa_feat"].shape[-3]
    device = feats["target_feat"].device

    # Controls whether the model uses in-place operations throughout
    # The dual condition accounts for activation checkpoints
    inplace_safe = False

    # Prep some features
    seq_mask = feats["seq_mask"]
    pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
    msa_mask = feats["msa_mask"]

    if model.globals.is_multimer:
        # Initialize the MSA and pair representations
        # m: [*, S_c, N, C_m]
        # z: [*, N, N, C_z]
        m, z = model.input_embedder(feats)
    elif model.seqemb_mode:
        # Initialize the SingleSeq and pair representations
        # m: [*, 1, N, C_m]
        # z: [*, N, N, C_z]
        m, z = model.input_embedder(
            feats["target_feat"],
            feats["residue_index"],
            feats["seq_embedding"]
        )
    else:
        # Initialize the MSA and pair representations
        # m: [*, S_c, N, C_m]
        # z: [*, N, N, C_z]
        m, z = model.input_embedder(
            feats["target_feat"],
            feats["residue_index"],
            feats["msa_feat"],
            inplace_safe=inplace_safe,
        )

    # Unpack the recycling embeddings. Removing them from the list allows 
    # them to be freed further down in this function, saving memory
    m_1_prev, z_prev, x_prev = reversed([prevs.pop() for _ in range(3)]) 

    # Initialize the recycling embeddings, if needs be 
    if None in [m_1_prev, z_prev, x_prev]:
        # [*, N, C_m]
        m_1_prev = m.new_zeros(
            (*batch_dims, n, model.config.input_embedder.c_m),
            requires_grad=False,
        )

        # [*, N, N, C_z]
        z_prev = z.new_zeros(
            (*batch_dims, n, n, model.config.input_embedder.c_z),
            requires_grad=False,
        )

        # [*, N, 3]
        x_prev = z.new_zeros(
            (*batch_dims, n, residue_constants.atom_type_num, 3),
            requires_grad=False,
        )

    pseudo_beta_x_prev = pseudo_beta_fn(
        feats["aatype"], x_prev, None
    ).to(dtype=z.dtype)

    # The recycling embedder is memory-intensive, so we offload first
    if model.globals.offload_inference and inplace_safe:
        m = m.cpu()
        z = z.cpu()

    # m_1_prev_emb: [*, N, C_m]
    # z_prev_emb: [*, N, N, C_z]
    m_1_prev_emb, z_prev_emb = model.recycling_embedder(
        m_1_prev,
        z_prev,
        pseudo_beta_x_prev,
        inplace_safe=inplace_safe,
    )

    del pseudo_beta_x_prev

    if model.globals.offload_inference and inplace_safe:
        m = m.to(m_1_prev_emb.device)
        z = z.to(z_prev.device)

    # [*, S_c, N, C_m]
    m[..., 0, :, :] += m_1_prev_emb

    # [*, N, N, C_z]
    z = add(z, z_prev_emb, inplace=inplace_safe)

    # Deletions like these become significant for inference with large N,
    # where they free unused tensors and remove references to others such
    # that they can be offloaded later
    del m_1_prev, z_prev, m_1_prev_emb, z_prev_emb

    # Embed the templates + merge with MSA/pair embeddings
    if model.config.template.enabled:
        template_feats = {
            k: v for k, v in feats.items() if k.startswith("template_")
        }

        template_embeds = model.embed_templates(
            template_feats,
            feats,
            z,
            pair_mask.to(dtype=z.dtype),
            no_batch_dims,
            inplace_safe=inplace_safe,
        )

        # [*, N, N, C_z]
        z = add(z,
                template_embeds.pop("template_pair_embedding"),
                inplace_safe,
                )

        if (
            "template_single_embedding" in template_embeds
        ):
            # [*, S = S_c + S_t, N, C_m]
            m = torch.cat(
                [m, template_embeds["template_single_embedding"]],
                dim=-3
            )

            # [*, S, N]
            if not model.globals.is_multimer:
                torsion_angles_mask = feats["template_torsion_angles_mask"]
                msa_mask = torch.cat(
                    [feats["msa_mask"], torsion_angles_mask[..., 2]],
                    dim=-2
                )
            else:
                msa_mask = torch.cat(
                    [feats["msa_mask"], template_embeds["template_mask"]],
                    dim=-2,
                )

    # Embed extra MSA features + merge with pairwise embeddings
    if model.config.extra_msa.enabled:
        if model.globals.is_multimer:
            extra_msa_fn = data_transforms_multimer.build_extra_msa_feat
        else:
            extra_msa_fn = build_extra_msa_feat

        # [*, S_e, N, C_e]
        extra_msa_feat = extra_msa_fn(feats).to(dtype=z.dtype)
        a = model.extra_msa_embedder(extra_msa_feat)

        if model.globals.offload_inference:
            # To allow the extra MSA stack (and later the evoformer) to
            # offload its inputs, we remove all references to them here
            input_tensors = [a, z]
            del a, z

            # [*, N, N, C_z]
            z = model.extra_msa_stack._forward_offload(
                input_tensors,
                msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                chunk_size=model.globals.chunk_size,
                use_deepspeed_evo_attention=model.globals.use_deepspeed_evo_attention,
                use_lma=model.globals.use_lma,
                pair_mask=pair_mask.to(dtype=m.dtype),
                _mask_trans=model.config._mask_trans,
            )

            del input_tensors
        else:
            # [*, N, N, C_z]
            z = model.extra_msa_stack(
                a, z,
                msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                chunk_size=model.globals.chunk_size,
                use_deepspeed_evo_attention=model.globals.use_deepspeed_evo_attention,
                use_lma=model.globals.use_lma,
                pair_mask=pair_mask.to(dtype=m.dtype),
                inplace_safe=inplace_safe,
                _mask_trans=model.config._mask_trans,
            )

    # Run MSA + pair embeddings through the trunk of the network
    # m: [*, S, N, C_m]
    # z: [*, N, N, C_z]
    # s: [*, N, C_s]          
    if model.globals.offload_inference:
        input_tensors = [m, z]
        del m, z
        m, z, s = model.evoformer._forward_offload(
            input_tensors,
            msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
            pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
            chunk_size=model.globals.chunk_size,
            use_deepspeed_evo_attention=model.globals.use_deepspeed_evo_attention,
            use_lma=model.globals.use_lma,
            _mask_trans=model.config._mask_trans,
        )

        del input_tensors
    else:
        m, z, s = model.evoformer(
            m,
            z,
            msa_mask=msa_mask.to(dtype=m.dtype),
            pair_mask=pair_mask.to(dtype=z.dtype),
            chunk_size=model.globals.chunk_size,
            use_deepspeed_evo_attention=model.globals.use_deepspeed_evo_attention,
            use_lma=model.globals.use_lma,
            use_flash=model.globals.use_flash,
            inplace_safe=inplace_safe,
            _mask_trans=model.config._mask_trans,
        )

    outputs["msa"] = m[..., :n_seq, :, :]
    outputs["pair"] = z
    outputs["single"] = s

    del z
    
    return outputs

def get_structure_module_embeddings(structure_module, 
                                    evoformer_output_dict,
                                    aatype,
                                    mask=None,
                                    inplace_safe=False,
                                    _offload_inference=False
                                    ):        
    """
    Args:
        evoformer_output_dict:
            Dictionary containing:
                "single":
                    [*, N_res, C_s] single representation
                "pair":
                    [*, N_res, N_res, C_z] pair representation
        aatype:
            [*, N_res] amino acid indices
        mask:
            Optional [*, N_res] sequence mask
    Returns:
        A dictionary of outputs
    """
    s = evoformer_output_dict["single"]

    if mask is None:
        # [*, N]
        mask = s.new_ones(s.shape[:-1])

    # [*, N, C_s]
    s = structure_module.layer_norm_s(s)

    # [*, N, N, C_z]
    z = structure_module.layer_norm_z(evoformer_output_dict["pair"])

    z_reference_list = None
    if (_offload_inference):
        assert (sys.getrefcount(evoformer_output_dict["pair"]) == 2)
        evoformer_output_dict["pair"] = evoformer_output_dict["pair"].cpu()
        z_reference_list = [z]
        z = None

    # [*, N, C_s]
    s_initial = s
    s = structure_module.linear_in(s)

    # [*, N]
    rigids = Rigid.identity(
        s.shape[:-1], 
        s.dtype, 
        s.device, 
        structure_module.training,
        fmt="quat",
    )
    outputs = []
    for i in range(structure_module.no_blocks):
        # [*, N, C_s]
        s = s + structure_module.ipa(
            s, 
            z, 
            rigids, 
            mask, 
            inplace_safe=inplace_safe,
            _offload_inference=_offload_inference, 
            _z_reference_list=z_reference_list
        )
        s = structure_module.ipa_dropout(s)
        s = structure_module.layer_norm_ipa(s)
        # get embeddings after step 7
        # if below line is uncommented --> get embeddings after step 9
        # s = structure_module.transition(s)              
        
    return s


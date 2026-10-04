import os
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
import numpy as np
def load_intervention_weights_loreft(reft_model, model_path,device,reft_config):
    file_list=os.listdir(model_path)
 
    for file in file_list:
        if file.endswith(".pt") and 'pytorch_model' not in file: 
            checkpoint_path = os.path.join(model_path, file)
            state_dict = torch.load(checkpoint_path, map_location=device)
            # Use the model's actual intervention key (format differs by component:
            # "layer_L_comp_block_output..." for LLaMA vs "comp_model_layers[L]_output..." for Qwen).
            module_key = list(reft_model.interventions.keys())[0]
            blk = reft_model.interventions[module_key].rotate_layer
           
            if is_parametrized(blk, "weight"):
                # leaves the current effective weight in blk.weight as a regular Parameter
                remove_parametrizations(blk, "weight", leave_parametrized=True)

            with torch.no_grad():
                src = state_dict["rotate_layer"]  # your source tensor
                blk.weight.data.copy_(src.to(dtype=blk.weight.dtype, device=blk.weight.device))

            
            reft_model.interventions[module_key].learned_source.weight.data.copy_(state_dict['weight'].float())
            reft_model.interventions[module_key].learned_source.bias.data.copy_(state_dict['bias'].float())
        

            reft_model.interventions[module_key].learned_source = reft_model.interventions[module_key].learned_source.float()
            

            print(f"Loaded weights for {module_key}.", f"shape: {reft_model.interventions[module_key].rotate_layer.weight.shape}")
        
    return reft_model


def yaml_defaults_from_argv(argv):
    """If --config <path.yaml> is in argv, load it and return a dict of defaults.

    Values are applied as argparse defaults (via parser.set_defaults), so any
    flag also passed on the command line still overrides the YAML. Returns {}
    when no --config is given.
    """
    import os
    import yaml
    path = None
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            path = argv[i + 1]
        elif a.startswith("--config="):
            path = a.split("=", 1)[1]
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"--config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def reinit_intervention_weights(reft_model):
    """
    Reinitialize all intervention weights in a ReFT model.
    
    For LowRankRotateLayer weights: Use orthogonal initialization
    For other weights (Linear layers, etc.): Use normal distribution
    
    Args:
        reft_model: A PyReFT model with interventions attribute
    
    Returns:
        reft_model: The model with reinitialized weights
    """
    for module_key, intervention in reft_model.interventions.items():
        print(f"Reinitializing weights for intervention: {module_key}")
        
        if hasattr(intervention, 'rotate_layer'):
            if hasattr(intervention.rotate_layer, 'parametrizations') and hasattr(intervention.rotate_layer.parametrizations, 'weight'):
                base_weight = intervention.rotate_layer.parametrizations.weight[0].base
                torch.nn.init.orthogonal_(base_weight)
                print(f"  - Orthogonally reinitialized rotate_layer with shape {base_weight.shape}")
            elif hasattr(intervention.rotate_layer, 'weight'):
                torch.nn.init.orthogonal_(intervention.rotate_layer.weight)
                print(f"  - Orthogonally reinitialized rotate_layer with shape {intervention.rotate_layer.weight.shape}")
        
        if hasattr(intervention, 'learned_source'):
            if hasattr(intervention.learned_source, 'weight'):
                intervention.learned_source.reset_parameters()
                print(f"  - Normal reinitialized learned_source.weight with shape {intervention.learned_source.weight.shape}")
            if hasattr(intervention.learned_source, 'bias') and intervention.learned_source.bias is not None:
                torch.nn.init.zeros_(intervention.learned_source.bias)
                print(f"  - Zero reinitialized learned_source.bias with shape {intervention.learned_source.bias.shape}")
        
        for name, param in intervention.named_parameters():
            if param.requires_grad:
                # Skip rotate_layer and learned_source as they're handled above
                if 'rotate_layer' not in name and 'learned_source' not in name:
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
                    print(f"  - Normal reinitialized {name} with shape {param.shape}")
    
    return reft_model


def compute_bert_score(reference, generated_text, rephrased_text, model_name='all-MiniLM-L6-v2'):

    model = SentenceTransformer(model_name)
    
    texts = [reference, generated_text, rephrased_text]
    embeddings = model.encode(texts, convert_to_tensor=True)
    
    ref_embedding = embeddings[0].unsqueeze(0)
    gen_embedding = embeddings[1].unsqueeze(0)
    rep_embedding = embeddings[2].unsqueeze(0)
    
    ref_vs_gen_similarity = util.cos_sim(ref_embedding, gen_embedding,).item()
    ref_vs_rep_similarity = util.cos_sim(ref_embedding, rep_embedding,).item()
    
    return {
        'reference_vs_generated': ref_vs_gen_similarity,
        'reference_vs_rephrased': ref_vs_rep_similarity
    }


def process_jsonl_with_bert_score(jsonl_path, output_path=None, model_name='all-MiniLM-L6-v2'):

    import json
    
    if output_path is None:
        output_path = jsonl_path.replace('.jsonl', '_with_bert_scores.jsonl')
    
    with open(jsonl_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        

        original_data = []
        rephrased_data = []
        for line in tqdm(infile):
            data = json.loads(line.strip())
            
            reference = data.get('reference', '')
            generated_text = data.get('generated_text', '')
            rephrased_text = data.get('rephrased_text', '')
            
            bert_scores = compute_bert_score(reference, generated_text, rephrased_text, model_name)
            data['bert_score_ref_vs_gen'] = bert_scores['reference_vs_generated']
            data['bert_score_ref_vs_rep'] = bert_scores['reference_vs_rephrased']
            original_data.append(bert_scores['reference_vs_generated'])
            rephrased_data.append(bert_scores['reference_vs_rephrased'])
            outfile.write(json.dumps(data) + '\n')

        print("Original data mean: ", np.mean(original_data))
        print("Rephrased data mean: ", np.mean(rephrased_data))
 
    return output_path
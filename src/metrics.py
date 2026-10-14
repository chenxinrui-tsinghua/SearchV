from tqdm import trange

import torch
import torch.nn.functional as F

from src.common_utils import to
from src.model_utils import Catcher, CatcherExit, get_layers, get_lm_head, get_lm_logits


def _move_llava_forward_batch(batch: dict, device: torch.device) -> dict:
    """Move Llava forward batch dict to device (same contract as searchv_evolution._move_forward_batch_to_device)."""
    out = {}
    for key, value in batch.items():
        if value is None:
            out[key] = None
        elif isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif isinstance(value, list):
            out[key] = [
                x.to(device) if isinstance(x, torch.Tensor) else x for x in value
            ]
        else:
            out[key] = value
    return out


@torch.no_grad()
def compute_perplexity(model, data, batch_size: int = 1):
    num_samples = len(data)
    device = next(model.parameters()).device
    # Running estimate of negative log-likelihood
    nll_running = 0
    # Number of tokens processed to far
    tokens_processed = 0
    # Loop through each batch
    for i in trange(0, num_samples, batch_size, desc="Computing perplexity", leave=False):
        j = min(i + batch_size, num_samples)
        inputs = torch.cat(data[i:j]).to(device)
        # Forward pass through the model
        lm_logits = model(inputs).logits
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        # Compute loss
        loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        # Calculate negative log likelihood
        a = shift_labels.numel() / (tokens_processed + shift_labels.numel())
        b = tokens_processed / (tokens_processed + shift_labels.numel())
        nll_running = a * loss + b * nll_running
        # Update number of processed tokens
        tokens_processed += shift_labels.numel()
    # Compute perplexity
    ppl = nll_running.exp().item()
    return ppl



@torch.no_grad()
def compute_kl_div(model, data, target_logits):
    num_samples = len(data)
    device = next(model.parameters()).device
    # Running sum of KL divergence
    kl_div_sum = 0.0
    # Number of tokens processed so far
    tokens_processed = 0
    # Loop through each sample
    pbar = trange(0, num_samples, desc="Computing KL fitness", leave=False)
    for idx in pbar:
        torch.cuda.empty_cache()
       
        inputs = data[idx]
        targets = target_logits[idx]
        # Forward pass through the model
        lm_logits = model(**inputs).logits

        # Both model and target logits represent predictions at each position
        # logits[i] = prediction for next token given context up to position i
        # We compare the same positions: both predicting the same next tokens
        # Exclude last position which has no next token to predict
        model_logits_aligned = lm_logits[:, :-1, :].contiguous()
        target_logits_aligned = targets[:, :-1, :].to(device).contiguous()
        
        # Compute KL divergence: KL(model || target)
        # Measures how model's predictions differ from target's predictions
        model_log_probs = model_logits_aligned.reshape(-1, model_logits_aligned.size(-1)).log_softmax(dim=-1)
        target_log_probs = target_logits_aligned.reshape(-1, target_logits_aligned.size(-1)).log_softmax(dim=-1)
        
        # Use batchmean: averages over tokens, sums over vocab
        kl_loss = F.kl_div(
            model_log_probs,
            target_log_probs,
            log_target=True,
            reduction="batchmean",
        )
        
        # Accumulate weighted average
        num_tokens = model_logits_aligned.shape[0] * model_logits_aligned.shape[1]
        kl_div_sum += kl_loss.item() * num_tokens
        tokens_processed += num_tokens
        
        torch.cuda.empty_cache()
    
    # Return average KL divergence per token
    return kl_div_sum / tokens_processed


@torch.no_grad()
def compute_gc_div(model, data, target_logits):
    """
    Global Contribution (GC): last-token KL aligned with SearchV LC calibration (see llava_llama LC loss).

    ``target_logits`` are baseline (no-skip) logits; may be full-sequence or last-position only
    ``[B, 1, V]`` (``[:, -1]`` is used either way).
    For each calib sample with seq_len > 1, uses F.kl_div(log_softmax(ref[:, -1]), softmax(cand[:, -1])).
    Same direction as LC (full ref vs ablated head); here ablation is the whole skip config in one pass.
    """
    num_samples = len(data)
    device = next(model.parameters()).device
    kl_sum = 0.0
    n = 0
    for idx in trange(0, num_samples, desc="Computing GC (SearchV last-token KL)", leave=False):
        torch.cuda.empty_cache()
        inputs = _move_llava_forward_batch(data[idx], device)
        ref_logits = target_logits[idx].to(device)
        cand_logits = model(**inputs).logits
        if cand_logits.shape[1] <= 1:
            continue
        # Match LC calibration: only accumulate when sequence length > 1
        kl = F.kl_div(
            F.log_softmax(ref_logits[:, -1].float(), dim=-1),
            F.softmax(cand_logits[:, -1].float(), dim=-1),
            reduction="batchmean",
        )
        kl_sum += kl.item()
        n += 1
        torch.cuda.empty_cache()
    return kl_sum / max(n, 1)


@torch.no_grad()
def compute_sparse_kl_div(model, data, target_logits):
    num_samples = len(data)
    device = next(model.parameters()).device
    # Running estimate of negative log-likelihood
    kl_div_running = 0
    # Number of tokens processed to far
    tokens_processed = 0
    # Loop through each batch
    for i in trange(0, num_samples, desc="Computing Sparse KL Divergence", leave=False):
        inputs = to(data[i], device=device)
        targets, target_ids = to(target_logits[i], device=device)
        # Forward pass through the model
        lm_logits = model(inputs).logits.gather(dim=-1, index=target_ids)
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_targets = targets[:, :-1, :]
        # Compute loss
        loss = F.kl_div(
            shift_logits.reshape(-1, shift_logits.size(-1)).log_softmax(dim=-1),
            shift_targets.reshape(-1, shift_targets.size(-1)).log_softmax(dim=-1),
            log_target=True,
            reduction="batchmean",
        )
        # Calculate negative log likelihood
        a = shift_targets.numel() / (tokens_processed + shift_targets.numel())
        b = tokens_processed / (tokens_processed + shift_targets.numel())
        kl_div_running = a * loss + b * kl_div_running
        # Update number of processed tokens
        tokens_processed += shift_targets.numel()
    return kl_div_running.item()


@torch.no_grad()
def compute_perplexity_layer_per_layer(
    model, data, device: torch.device = "cuda", offload: bool = False, batch_size: int = 1
):
    num_samples = len(data)

    # Get layers
    input_embeddings = model.get_input_embeddings()
    layers = get_layers(model)
    lm_head = get_lm_head(model)

    # Process input embeddings
    input_embeddings = input_embeddings.to(device)
    layers[0] = layers[0].to(device)
    layers[0] = Catcher(layers[0], offload=offload)
    for i in range(0, num_samples, batch_size):
        try:
            j = min(i + batch_size, num_samples)
            input_ids = torch.cat(data[i:j], dim=0).to(device)
            # call model.forward to trigger the Catcher
            model(input_ids, attention_mask=torch.ones_like(input_ids))
        except CatcherExit:
            pass
    inputs = layers[0].inputs
    input_kwargs = layers[0].input_kwargs
    layers[0] = layers[0].module
    input_embeddings = input_embeddings.cpu()

    # Process layers
    for layer_id in trange(len(layers), desc="Processing evaluation data layer-by-layer", leave=False):
        layer = layers[layer_id].to(device)
        for i, (inps, inp_kwargs) in enumerate(zip(inputs, input_kwargs)):
            out = layer(inps.to(device), **inp_kwargs)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if offload:
                out = out.cpu()
            inputs[i] = out
        # Offload layer
        layers[layer_id] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    # Compute perplexity
    lm_head = lm_head.to(device)
    # Running estimate of negative log-likelihood
    nll_running = 0
    # Number of tokens processed to far
    tokens_processed = 0
    for i, inps in zip(range(0, num_samples, batch_size), inputs):
        # Get input_ids
        j = min(i + batch_size, num_samples)
        input_ids = torch.cat(data[i:j], dim=0).to(device)
        # Forward pass through the model
        lm_logits = get_lm_logits(inps.to(device), model)
        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:]
        # Compute loss
        loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        # Calculate negative log likelihood
        a = shift_labels.numel() / (tokens_processed + shift_labels.numel())
        b = tokens_processed / (tokens_processed + shift_labels.numel())
        nll_running = a * loss + b * nll_running
        # Update number of processed tokens
        tokens_processed += shift_labels.numel()
    # put lm_head back to original device
    lm_head = lm_head.cpu()
    # Compute perplexity
    ppl = nll_running.exp().item()
    return ppl

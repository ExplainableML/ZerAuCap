import sys
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import argparse
import random
import pandas as pd


def parse_prompt(text):
    """
    process the prompt text;
    """
    eos_token = "<|endoftext|>"
    text = text.strip(eos_token).strip()
    left_bracket_idx, right_bracket_idx = -1, -1
    for idx in range(len(text)):
        char = text[idx]
        if char == "[" and left_bracket_idx == -1:  # first [ is met
            left_bracket_idx = idx
        elif char == "]" and right_bracket_idx == -1:  # first ] is met
            right_bracket_idx = idx
        else:
            pass
    res_text = ""
    remove = False
    if left_bracket_idx > -1 and right_bracket_idx > left_bracket_idx:
        if right_bracket_idx - left_bracket_idx <= 6:
            remove = True
        else:
            pass

    for idx in range(len(text)):
        if remove:
            if idx >= left_bracket_idx and idx <= right_bracket_idx:
                continue
            else:
                res_text += text[idx]
        else:
            res_text += text[idx]
    res_text = res_text.strip()
    res_text = " ".join(res_text.split()).strip()
    return res_text


def typical_filtering(scores, mass, min_tokens_to_keep, filter_value):
    # calculate entropy
    normalized = torch.nn.functional.log_softmax(scores, dim=-1)
    p = torch.exp(normalized)
    ent = -(normalized * p).nansum(-1, keepdim=True)

    # shift and sort
    shifted_scores = torch.abs((-normalized) - ent)
    sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
    sorted_logits = scores.gather(-1, sorted_indices)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative mass above the threshold
    last_ind = (cumulative_probs < mass).sum(dim=1)
    last_ind[last_ind < 0] = 0
    sorted_indices_to_remove = sorted_scores > sorted_scores.gather(
        1, last_ind.view(-1, 1)
    )
    if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )

    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores


def top_k_top_p_filtering(
    logits, top_k=0, top_p=0.0, threshold=-float("Inf"), filter_value=-np.inf
):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    return logits


# ========== batch version ========= #
def ranking_fast(context_hidden, next_hidden, next_top_k_probs, alpha, beam_width):
    """
    context_hidden: bsz*beam x seqlen x embed_dim
    next_hidden: bsz*beam x 1 x embed_dim
    next_top_k_probs: bsz x beam
    """
    _, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(
        norm_context_hidden, norm_next_hidden.transpose(1, 2)
    ).squeeze(-1)  # [B*K, S]
    scores, _ = torch.max(cosine_matrix, dim=-1)  # [B*K]
    next_top_k_probs = next_top_k_probs.view(-1)  # [B*K]
    scores = (1.0 - alpha) * next_top_k_probs - alpha * scores
    scores = torch.stack(torch.split(scores, beam_width))  # [B, K]
    selected_idx = scores.max(dim=-1)[1]  # [B]
    return selected_idx


def ContrastiveDecodingOneStepFast(
    model,
    ids,
    beam_width,
    alpha,
    past_key_values,
    last_hidden_states,
    vocab,
    logit_for_next_step,
    first_step=False,
):
    # input_ids: [B, S]
    if first_step:
        output = model(
            input_ids=ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = output.past_key_values
        last_hidden_states = output.hidden_states[-1]  # [B, S, E]
        logit_for_next_step = output.logits[:, -1, :]  # [B, V]
    bsz, seqlen, embed_dim = last_hidden_states.size()
    p = random.uniform(0, 1)

    next_probs = F.softmax(logit_for_next_step, dim=-1)
    _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)  # [B, K]
    top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)  # [B, K]
    # compute new hidden
    past_key_values = enlarge_past_key_values(past_key_values, beam_width)
    output = model(
        input_ids=top_k_ids.view(-1, 1),
        attention_mask=torch.ones_like(top_k_ids.view(-1, 1)),
        past_key_values=past_key_values,
        output_hidden_states=True,
        use_cache=True,
    )
    past_key_values = output.past_key_values
    logits = output.logits[:, -1, :]  # [B*K, V]
    next_hidden = output.hidden_states[-1]  # [B*K, 1, E]
    context_hidden = (
        last_hidden_states.unsqueeze(1)
        .expand(-1, beam_width, -1, -1)
        .reshape(bsz * beam_width, seqlen, embed_dim)
    )  # [B*K, S, E]

    selected_idx = ranking_fast(
        context_hidden,
        next_hidden,
        top_k_probs,  # [B, K]
        alpha,
        beam_width,
    )  # [B]
    # prepare for the next step
    next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)  # [B, 1]
    next_hidden = torch.stack(
        torch.split(next_hidden.squeeze(dim=1), beam_width)
    )  # [B, K, E]
    next_hidden = next_hidden[range(bsz), selected_idx, :]  # [B, E]
    last_hidden_states = torch.cat(
        [last_hidden_states, next_hidden.unsqueeze(1)], dim=1
    )  # [B, S, E]
    past_key_values = select_past_key_values(past_key_values, beam_width, selected_idx)
    logits = torch.stack(torch.split(logits, beam_width))[
        range(bsz), selected_idx, :
    ]  # [B, V]
    # next_id: [B, 1]
    return next_id, past_key_values, last_hidden_states, logits


def enlarge_past_key_values(past_key_values, beam_width):
    # from [B, num_head, seq_len, esz] to [B*K, num_head, seq_len, esz]
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            # item is the key and value matrix
            bsz, num_head, seq_len, esz = item.size()
            item = (
                item.unsqueeze(1)
                .expand(-1, beam_width, -1, -1, -1)
                .reshape(bsz * beam_width, num_head, seq_len, esz)
            )  # [bsz*beam, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values


def select_past_key_values(past_key_values, beam_width, selected_idx):
    """select_idx: [B]"""
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            bsz_and_beam, num_head, seq_len, esz = item.size()
            bsz = int(bsz_and_beam // beam_width)
            item = torch.stack(
                torch.split(item, beam_width, dim=0)
            )  # [B, K, num_head, seq_len, esz]
            item = item[
                range(bsz), selected_idx, :, :, :
            ]  # [B, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values


# ========== fast plug and play version ========= #
def plug_and_play_fast_ranking(
    context_hidden,
    next_hidden,
    next_top_k_ids,
    next_top_k_probs,
    alpha,
    beta,
    batch_class_score,
    beam_width,  # choose k (=beam_width) candidates for being our best word
    top_k_ids,
    tokenizer,
    step,
    end_penalty,
):
    """
    context_hidden: beam_width x context_len x embed_dim   ; for each word in every candidate's context get embedding vector
    next_hidden: beam_width x 1 x embed_dim        ; current candidate's embedding vector
    next_top_k_ids: beam_width x 1      ; top k candidates
    batch_class_score: beam_width x 1   ; magic score of every k candidate
    """

    _, context_len, embed_dim = context_hidden.size()
    # print(context_len, embed_dim)
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(
        norm_context_hidden, norm_next_hidden.transpose(1, 2)
    ).squeeze(-1)
    scores, _ = torch.max(cosine_matrix, dim=-1)  # degeneration penalty
    next_top_k_probs = next_top_k_probs.view(-1)
    # print("model_confidence: ")
    # print(next_top_k_probs)
    # print("degeneration pen: ")
    # print(scores)
    # print("magic score: ")
    # print(batch_class_score.view([beam_width]))
    # print("sum of magic scores: ")
    # print(torch.sum(batch_class_score.view([beam_width])))

    scores = (
        (1.0 - alpha) * next_top_k_probs
        - alpha * scores
        + beta * batch_class_score.view([beam_width])
    )
    scores = torch.stack(torch.split(scores, beam_width)).squeeze()

    # Many tokens (high step), low penalty. Few tokens (low step), high penalty for EOS token or break tokens
    # check if token is in list of [EOS, BREAK TOKENS]; if yes, decrease score!
    scores_and_indices = scores.topk(next_top_k_ids.shape[1])

    top_k_tokens = torch.tensor(
        [top_k_ids.squeeze()[index] for index in scores_and_indices.indices.squeeze()],
        device="cuda",
    )
    tokens_to_penalize = (
        tokenizer.encode(". ! ?", return_tensors="pt").cuda().squeeze()
    )  # includes sos and eos in gpt2 and opt
    base_penalty = 0
    sequential_penalty = end_penalty
    penalty = (
        base_penalty + sequential_penalty * step
    )  # penalty is a function of the step. The lower the step, the higher the penalty

    penalized_scores = []
    for k, token in enumerate(top_k_tokens):
        if token in tokens_to_penalize:
            # penalized_scores[k] = scores[k] * penalty
            penalized_scores.append(scores[k] * penalty)

        else:
            penalized_scores.append(scores[k])

    penalized_scores = torch.tensor(penalized_scores, device="cuda")

    selected_idx = penalized_scores.max(dim=-1)[1]

    # save unsoftmaxed cos sim of every iteration and the respective untokenized bits

    # print("Winner token: ")
    # print(selected_idx)

    # RETURN both var of batch_class_score and next_top_k_probs

    return selected_idx


def PlugAndPlayContrastiveDecodingOneStepFast(
    model,
    input_ids,
    prefix_len,
    beam_width,
    alpha,
    beta,
    simctg_tokenizer,
    image_embeds,
    clip,
    clip_text_max_len,
    past_key_values,
    last_hidden_states,
    logit_for_next_step,
    include_prompt_magic,
    step,
    end_penalty,
    first_step=False,
    input_ids_for_class=None,
):  # , add_token_level_score=False):
    """
    model: the generation model, e.g., gpt2
    input_ids: 1 x seqlen
    """

    if first_step:
        output = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )  # breakpoint
        past_key_values = (
            output.past_key_values
        )  # previously computed key/value attention pair
        last_hidden_states = output.hidden_states[
            -1
        ]  # [B, S, E] get last hidden state before logit
        logit_for_next_step = output.logits[
            :, -1, :
        ]  # [B, V] 1 x vocabulary size (every sample gets logits for every word) (get last logit vector)

    bsz, seqlen, embed_dim = (
        last_hidden_states.size()
    )  # at testing: batch size 1, every word has one embedding vector (on first step: 6 embedding vecs)

    next_probs = F.softmax(logit_for_next_step, dim=-1)
    _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)
    top_k_probs = torch.gather(
        next_probs, dim=1, index=top_k_ids
    )  # get probabilities for top k words (model confidence)

    # compute the new hidden ### HERE  ##
    past_key_values = enlarge_past_key_values(past_key_values, beam_width)

    if "OPTForCausalLM" in str(type(model)):
        attention_mask = torch.ones(beam_width, 1 + seqlen, device="cuda")

    else:
        attention_mask = torch.ones_like(top_k_ids.view(-1, 1))

    output = model(
        input_ids=top_k_ids.view(-1, 1),
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        output_hidden_states=True,
        use_cache=True,
    )
    past_key_values = output.past_key_values
    logits = output.logits[:, -1, :]
    next_hidden = output.hidden_states[-1]
    context_hidden = (
        last_hidden_states.unsqueeze(1)
        .expand(-1, beam_width, -1, -1)
        .reshape(bsz * beam_width, seqlen, embed_dim)
    )

    # prepare for the classification model
    input_ids_for_class_ = torch.cat(
        [
            input_ids_for_class.unsqueeze(1)
            .expand(-1, beam_width, -1)
            .reshape(bsz * beam_width, seqlen),
            top_k_ids.view(-1, 1),
        ],
        dim=-1,
    )  # dim: k x n_prompt_tokens + candidate token     for every top k candidate prompt tokens and its token

    batch_text_list = []

    if include_prompt_magic == "True":
        # magic score is computed for the prompt + proposed_word
        for one_input_id in input_ids_for_class_:
            one_text = simctg_tokenizer.decode(one_input_id)
            batch_text_list.append(one_text)

    else:
        for one_input_id in input_ids_for_class_:
            one_text = simctg_tokenizer.decode(
                one_input_id[prefix_len:][-clip_text_max_len:]
            )
            """
            decode: convert token ids to string
            only decode the tokens after the prompt. on the first step, the candidate token only. 
            on the second step, the first generated token and the new candidate, ... (all up to the defined max len)
            """

            # we only consider the class score of the generated text continuation
            batch_text_list.append(one_text)

    with torch.no_grad():
        if "AudioCLIP" in str(type(clip)):
            batch_text_list_ = [[candidate] for candidate in batch_text_list]
            text_embeds = clip.encode_text(batch_text_list_)

        elif "CLAP" in str(type(clip)):
            text_embeds = clip.encode_text(batch_text_list, use_tensor=True).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        else:
            text_embeds = clip.encode_text(batch_text_list)

    scaled_cos_sim = clip.logit_scale_a * torch.cosine_similarity(
        image_embeds, text_embeds
    )
    scaled_cos_sim = torch.unsqueeze(scaled_cos_sim.t(), 0)
    batch_score = scaled_cos_sim.softmax(dim=-1)

    # does CLAP get normalized?

    selected_idx = plug_and_play_fast_ranking(  # does beam search
        context_hidden,
        next_hidden,
        top_k_ids,
        top_k_probs,
        alpha,
        beta,
        batch_score,
        beam_width,
        top_k_ids,
        simctg_tokenizer,
        step,
        end_penalty,
    )

    # prepare for the next step
    next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)
    next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width))
    next_hidden = next_hidden[range(bsz), selected_idx, :]
    last_hidden_states = torch.cat(
        [last_hidden_states, next_hidden.unsqueeze(1)], dim=1
    )
    past_key_values = select_past_key_values(past_key_values, beam_width, selected_idx)
    logits = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :]
    input_ids_for_class = torch.cat([input_ids_for_class, next_id], dim=-1)
    return (
        next_id,
        past_key_values,
        last_hidden_states,
        logits,
        input_ids_for_class,
    )  # , cos_sims_every_word

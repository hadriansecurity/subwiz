"""
This code is largely copied from the nanogpt repository by Andrej Karpathy.
The generate method has been modified to output many sequences.

https://github.com/karpathy/nanoGPT

Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import re
from datetime import datetime
import math
import inspect
import os
import uuid

from pydantic import BaseModel, ConfigDict
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedTokenizerFast
from typing import Callable, Optional


VALID_SUBDOMAIN_RE = re.compile(
    r"^(?!-)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)(?:\.([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?))*(?<!\.)$",
    re.IGNORECASE,
)

VALID_SUBDOMAIN_START_RE = re.compile(
    r"^(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61})?)?$",
    re.IGNORECASE,
)


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTConfig(BaseModel):
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )
    tokenizer_file: str = "resources/tokenizer.json"

    model_config = ConfigDict(extra="ignore")


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=config.tokenizer_file, clean_up_tokenization_spaces=True
        )
        self.end_token = self.tokenizer("[END]")["input_ids"][0]
        self.comma_token = self.tokenizer(",")["input_ids"][0]
        self.delim_token = self.tokenizer("[DELIM]")["input_ids"][0]

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        # print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # with torch.autograd.detect_anomaly():
        #     if torch.isnan(idx).any():
        #         print(f'NAN found!: {idx}')

        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @property
    def device(self) -> str:
        # assign model inputs to the right device
        return next(self.lm_head.parameters()).device.type

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 12,
        temperature: float = 0.0,
        topn: int = 100,
        pruning_ratio: float = 4,
        pruning_offset: float = 5,
        log_file: Optional[str] = None,
        on_iteration: Callable = None,
        blocked_outputs: set[str] = None,  # doesn't output these strings
    ) -> torch.Tensor:
        """Custom generate function that outputs topn sequences, different to the original nanoGPT implementation."""

        if topn <= 0:
            raise ValueError("topn should be greater than 0")

        if not 0 < max_new_tokens <= 20:
            raise ValueError("max_new_tokens should be in (0, 20]")

        run_uuid = uuid.uuid4()

        idx = idx.to(self.device)
        sequences = idx.unsqueeze(0)

        probabilities = torch.tensor([1.0], device=self.device)

        finished_sequences = torch.tensor([], device=self.device)
        finished_probs = torch.tensor([], device=self.device)

        # compute number of sequences to pass to each iteration
        sequences_per_iter = round(pruning_offset + topn / pruning_ratio)

        for i in range(max_new_tokens):
            if on_iteration is not None:
                on_iteration()

            # trim the sequences down to block size
            sequences = sequences[:, -self.config.block_size :]

            # remove any invalid subdomain starts
            outputs = self.tokenizer.batch_decode(sequences[:, -i:])
            outputs = [o.replace(" ", "") for o in outputs]
            outputs = [o.rsplit("[DELIM]", 1)[-1] for o in outputs]
            valid_start_mask = [
                bool(VALID_SUBDOMAIN_START_RE.match(o)) for o in outputs
            ]
            sequences = sequences[valid_start_mask]
            probabilities = probabilities[valid_start_mask]
            outputs = [o for o, valid in zip(outputs, valid_start_mask) if valid]

            # inference the model in batches
            batch_size = 8
            logits, _ = self(sequences[:batch_size])
            for j in range(batch_size, len(sequences), batch_size):
                new_logits, _ = self(sequences[j : j + batch_size])
                logits = torch.cat(tensors=(new_logits, logits), dim=0)
            logits = logits.squeeze(1)

            # take N most probable next tokens for each sequence
            output_probs = F.softmax(logits, dim=-1)
            new_sequence_probs = output_probs * probabilities.unsqueeze(1)

            # remove finished sequences (after end token) and cache their probs
            if i > 0:
                # likelihood of each sequence having an end or comma token next
                comma_token_probs = new_sequence_probs[:, self.comma_token]
                end_token_probs = new_sequence_probs[:, self.end_token]
                _finish_probs = end_token_probs + comma_token_probs
                _finished_sequences = sequences.clone().detach()

                # don't output any invalid subdomains
                valid_subdomain_mask = [
                    bool(VALID_SUBDOMAIN_RE.match(o)) for o in outputs
                ]
                _finish_probs = _finish_probs[valid_subdomain_mask]
                _finished_sequences = _finished_sequences[valid_subdomain_mask]
                outputs = [o for o, val in zip(outputs, valid_subdomain_mask) if val]

                # don't output anything in blocked_outputs
                if blocked_outputs and outputs:
                    blocked_outputs_mask = torch.tensor(
                        [s not in blocked_outputs for s in outputs], device=self.device
                    )
                    _finish_probs = _finish_probs[blocked_outputs_mask]
                    _finished_sequences = _finished_sequences[blocked_outputs_mask]

                # add all sequences to the finished_sequences with prob of ending
                finished_sequences = torch.cat(
                    (finished_sequences, _finished_sequences)
                )
                finished_probs = torch.cat((finished_probs, _finish_probs), dim=-1)

            # remove sequences and tokens with a probability that is too low
            if len(finished_sequences) > topn:
                # torch.kthvalue is not implemented on MPS, so we use topk
                lowest_viable_probability = torch.topk(finished_probs, topn).values[-1]
                viable_sequences = probabilities >= lowest_viable_probability

                if viable_sequences.sum() == 0:
                    break

                # remove sequences with a too low probability
                sequences = sequences[viable_sequences]
                probabilities = probabilities[viable_sequences]
                logits = logits[viable_sequences]
                new_sequence_probs = new_sequence_probs[viable_sequences]

                # remove tokens that would generate sequences with too low probability
                token_mask = new_sequence_probs < lowest_viable_probability
                if token_mask.sum() == 0:
                    break

                new_sequence_probs[token_mask] = 0
                logits[token_mask] = 0

            # do not sample the end token or comma token for the next iter
            new_sequence_probs[:, self.end_token] = 0
            new_sequence_probs[:, self.comma_token] = 0

            # number of sequences to pass to next iteration
            num_nonzero_probs = torch.count_nonzero(new_sequence_probs).item()
            num_seqs_next_iter = min(sequences_per_iter, num_nonzero_probs)

            if num_seqs_next_iter == 0:
                break

            if temperature == 0:  # select most likely tokens for next iteration
                new_sequence_probs = new_sequence_probs.flatten()
                _, idx_next = torch.topk(new_sequence_probs, num_seqs_next_iter)

            else:  # sample tokens for next iteration
                # recalculate probabilities using temperature
                scaled_logits = logits / (temperature + 1e-1)
                probs_with_temp = F.softmax(scaled_logits, dim=-1)
                probs_with_temp = probs_with_temp * probabilities.unsqueeze(1)

                probs_with_temp[:, self.end_token] = 0
                probs_with_temp[:, self.comma_token] = 0

                # sample tokens for next iteration
                probs_with_temp = probs_with_temp.flatten()
                probs_with_temp[probs_with_temp < 0] = 0
                idx_next = torch.multinomial(probs_with_temp, num_seqs_next_iter)

            # add the sampled tokens to the end of each sequence
            sequence_idx = idx_next // self.config.vocab_size
            token_values = idx_next % self.config.vocab_size

            sequences = sequences[sequence_idx]
            sequences = torch.cat([sequences, token_values.unsqueeze(1)], dim=-1)
            probabilities = new_sequence_probs.flatten()[idx_next]

            if log_file is not None:
                _, current_best_idx = torch.topk(
                    finished_probs, min(topn, len(finished_probs))
                )
                current_best = finished_sequences[current_best_idx]
                self.log_generation_data(
                    log_file=log_file,
                    run_id=run_uuid,
                    topn=topn,
                    x=idx,
                    iteration=i,
                    probabilities=probabilities,
                    current_preds=current_best,
                    finished_probs=finished_probs,
                )

        if len(finished_sequences) <= topn:
            return finished_sequences

        _, final_indices = torch.topk(finished_probs, topn)
        final_sequences = finished_sequences[final_indices]

        return final_sequences

    @staticmethod
    def from_checkpoint(
        path: str,
        return_train_params: bool = False,
        device: str = "cpu",
        tokenizer_path: Optional[str] = None,
    ):
        checkpoint = torch.load(path, map_location=device, weights_only=True)

        config = GPTConfig(**checkpoint["model_args"])
        if tokenizer_path:
            config.tokenizer_file = tokenizer_path
        model = GPT(config)
        state_dict = checkpoint["model"]

        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.to(device)

        if not return_train_params:
            return model

        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        optim_state = checkpoint["optimizer"]

        assert isinstance(iter_num, int)
        assert isinstance(best_val_loss, torch.Tensor)
        assert isinstance(optim_state, dict)

        return model, iter_num, best_val_loss, optim_state

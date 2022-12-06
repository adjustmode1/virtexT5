r"""
This Beam Search implementation is adapted with minor modifications from
`AllenNLP <https://github.com/allenai/allennlp/blob/master/allennlp/nn/beam_search.py>`_.

Thanks to the developers of AllenNLP!

**Update (v1.2):** The "backpointer" trick in Beam Search (as implemented in
AllenNLP) does not work well with autoregressive models (transformers). It is
now removed and it improves qualitative predictions and captioning metrics
(CIDEr/SPICE) for VirTex. Updated captioning results are on ArXiv v3. Refer
`CHANGELOG <https://github.com/kdexd/virtex/blob/master/CHANGELOG.md>`_ and
`Release Page <https://github.com/kdexd/virtex/releases/tag/v1.2>`_ for more
details.

Huge thanks to Nicolas Carion (@alcinos) and Aishwarya Kamath (@ashkamath) for
helping me fix this bug!
"""
from typing import Callable, Tuple
import warnings

import torch
from torch.nn import functional as F


class AutoRegressiveBeamSearch:
    r"""
    Implements the beam search algorithm for decoding the most likely captions.

    Args:
        eos_index: The index of the end token (``[EOS]``) in vocabulary.
        max_steps: The maximum number of decoding steps.
        beam_size: The width of the beam used.
        per_node_beam_size: The maximum number of candidates to consider per node,
            at each step in the search. Setting this parameter to a number smaller
            than ``beam_size`` may give better results, as it can introduce more
            diversity into the search. See `Beam Search Strategies for Neural
            Machine Translation. Freitag and Al-Onaizan, 2017 <https://arxiv.org/abs/1702.01806>`_.
    """

    def __init__(
        self,
        eos_index: int = 3,
        max_steps: int = 64,
        beam_size: int = 7,
        per_node_beam_size: int = 2,
        sos_index: int = 2,
        alpha: int = 0.7
    ) -> None:
        self.eos_index = eos_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size
        self.alpha = alpha
        self.sos_index = sos_index
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def search(
        self,
        visual_features: torch.Tensor,
        textual: any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
            visual_features = visual_features[None, ...]
            memory = textual.encode(visual_features.to(self.device))
            target = torch.tensor([[self.eos_index]]) # BOS bằng mấy
            with torch.no_grad():
                output = textual.decode(target.to(self.device), memory)
                logits = textual.generator(output.last_hidden_state)
            scaled_logits = torch.log_softmax(logits[None,:], dim=1).cpu().squeeze(0) # over vocab size 
            weights, candidates = torch.topk(input=scaled_logits, k=self.beam_size, largest=True)
            response_tracker = []  # for valid final sequence 
            sequence_tracker = []  # for current active sequence
            print(candidates)
            for idx in candidates[0][0]:
                print("idx: ",idx)
                option = torch.tensor([[idx]])  # a new option into the search tree 
                sequence = torch.cat([target, option], dim=1)
                sequence_tracker.append(sequence)
            print("sequence_tracker: ",sequence_tracker)
            keep_generating = True 
            while keep_generating:
                input_batch = torch.vstack(sequence_tracker)
                print("input_batch: ",input_batch)
                for x in input_batch:
                  print("element:",torch.tensor([[x[-1]]]))
                  with torch.no_grad():
                      input_memory = memory.repeat(1, 1, 1)
                      output = textual.decode(torch.tensor([[x[-1]]]).to(self.device), input_memory)
                      logits = textual.generator(output.last_hidden_state)    
 
                  scaled_logits = torch.log_softmax(logits[None,:], dim=1).cpu().squeeze(0)
                  # bị cắt
                  length = input_batch.shape[1] # input_batch
                  vocab_size = scaled_logits.shape[1] # scaled_logits
                  print("scaled_logits",scaled_logits.shape)
                  print("weight: ",weights.shape)
                  weighted_logits = (scaled_logits + weights[0,-1,:][:, None]) / length ** self.alpha  
                  weights, candidates = torch.topk(input=weighted_logits, k=self.beam_size, largest=True) # beam_width
                  weights = weights * length ** self.alpha  # denormalize
                  print("weights",weights.shape)
                  weights_tmp = []
                  sequence_tmp = []
                  print("result:",candidates)
                  for idx, pos in enumerate(candidates):
                      row = torch.div(pos, vocab_size, rounding_mode='floor') # get relative position over nb_sequences 
                      col = pos % vocab_size  # get relative position over vocab_size 
                      sequence = torch.cat([sequence_tracker[row], torch.tensor([[col]])], dim=1)
                      if col == self.eos_index:
                          flattened_sequence = torch.flatten(sequence).tolist()
                          sequence_score = weights[idx] / len(flattened_sequence) ** self.alpha
                          response_tracker.append((flattened_sequence, sequence_score))  # a sentence was built ##### response_tracker
                          if len(response_tracker) == self.beam_size:
                              keep_generating = False 
                              break  # end the for loop over candidates
                      elif sequence.shape[1] < self.max_steps - 1:
                          weights_tmp.append(weights[row])
                          sequence_tmp.append(sequence)
                  # end for loop over candidates ...!
                  if len(sequence_tmp) == 0: 
                      keep_generating = False 
                      break
                  else:               
                      weights = torch.tensor(weights_tmp)
                      sequence_tracker = sequence_tmp
            return response_tracker
        # end while search loop ...! 

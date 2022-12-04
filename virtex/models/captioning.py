import copy
import functools
from typing import Any, Dict

import torch
import numpy as np
from torch import nn

from virtex.data.tokenizers import SentencePieceBPETokenizer
from virtex.modules.textual_heads import TextualHead
from virtex.modules.visual_backbones import VisualBackbone
import pickle as pk
import sentencepiece as spm
def build_mask(seq):
    seq_length = seq.shape[1]
    mask = np.fromfunction(lambda i,j: j > i, shape=(seq_length, seq_length))
    return torch.as_tensor(mask) 

def build_key_padding_mask(seq, pad_idx):
    seq_key_padding_mask = (seq == pad_idx)
    return seq_key_padding_mask

class CaptioningModel(nn.Module):
    r"""
    A model to perform image captioning (in both forward and backward directions
    independently, only in forward direction). nó bao gồm một
    :class:`~virtex.modules.visual_backbones.VisualBackbone` và một
    :class:`~virtex.modules.textual_heads.TextualHead` on top of it.

    trong quá trình train, nó tối đa khả năng của một caption đúng điều kiện dựa trên
    các feature hình ảnh. trong quá trình suy luận, nó dự đoán 1 caption cho
    một hình ảnh đầu vào thông qua beam search decoding.

    Args:
        visual: A :class:`~virtex.modules.visual_backbones.VisualBackbone` mà
            tính toán visual features từ hình ảnh đầu vào
        textual: A :class:`~virtex.modules.textual_heads.TextualHead` which
            đưa ra các dự đoán cuối cùng dựa trên các visual features.
        sos_index:vị trí bắt đầu của token (``[SOS]``) trong vocabulary.
        eos_index: vị trí cuối của token (``[EOS]``) trong vocabulary.
        caption_backward: Whether to *also* perform captioning in backward
            direction. mặc định là ``False`` -- chỉ forward captioning is
            performed. khi có giá trị là ``True``, tạo ra 1 clone textual head, nó
            không chỉ chia sẻ weights với mô hình "forward" ngoại trừ input/output embeddings.
        decoder: A :class:`~virtex.utils.beam_search.AutoRegressiveBeamSearch`
            or :class:`~virtex.utils.nucleus_sampling.AutoRegressiveNucleusSampling`
            object for decoding captions during inference (không sử dụng trong quá trình training).
    """

    def __init__(
        self,
        visual: VisualBackbone,
        textual: TextualHead,
        caption_backward: bool = False,
        sos_index: int = 1,
        eos_index: int = 2,
        decoder: Any = None,
    ):
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.padding_idx = self.textual.padding_idx
        self.caption_backward = caption_backward
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Clone the textual module for backward direction if doing captioning
        # in both directions (separately).
        if self.caption_backward:
            self.backward_textual = copy.deepcopy(self.textual)

            # Share weights for visual projection, and input/output embeddings.
            self.backward_textual.position_encoder = self.textual.position_encoder
            self.backward_textual.adaptaror = self.textual.adaptaror
            self.backward_textual.token_embedder = self.textual.token_embedder
            self.backward_textual.transformer = self.textual.transformer
            self.backward_textual.generator = self.textual.generator
        # These boundary indices are needed for beam search.
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.decoder = decoder
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        r"""
        cho 1 batch hình ảnh và caption, tính toán ghi lại khả năng xẩy ra loss mỗi
        caption token trong quá trình training. trong quá trình suy luận (with images), dự đoán
        một caption thông qua 1 trong 2 beam search decoding hoặc nucleus sampling.

        Args:
            batch: A batch of images and (optionally) ground truth caption tokens.
                dạng có thể có của set of keys: ``{"image_id", "image", "caption_tokens",
                "noitpac_tokens", "caption_lengths"}``.

        Returns:
            1 dict với cấu trúc sau, chứa loss để optimization,
            loss components để log directly to tensorboard, và optionally
            predictions.

            .. code-block::

                {
                    "loss": torch.Tensor,
                    "loss_components": {
                        "captioning_forward": torch.Tensor,
                        "captioning_backward": torch.Tensor, (optional)
                    },
                    "predictions": torch.Tensor
                }
        """

        # shape: (batch_size, channels, height, width)
        visual_features = batch["image"]
        # đặc điểm của visual là gì
        #mới
        batch_size = visual_features.shape[1] # batch size = 1
        #end mới
        if "caption_tokens" in batch:
            caption_tokens = batch["caption_tokens"]
            caption_lengths = batch["caption_lengths"]

            # shape: (batch_size, max_caption_length, vocab_size)
            tgt_input = caption_tokens[:, :-1]
            tgt_output = caption_tokens[:, 1:]
            tgt_mask = build_mask(tgt_input).to(self.device) # chuyển về ma trận true false (j>i)
            tgt_key_padding_mask = build_key_padding_mask(tgt_input, self.padding_idx).to(self.device) # seq có bằng pad_idx không lưu vào ma trận
            memory = self.textual.encode(src=visual_features.to(self.device)) # đưa dữ liệu vào encode và nhận được memory là ma trận 

            output = self.textual.decode(
                tgt=tgt_input.to(self.device), 
                memory=memory, 
                tgt_mask=tgt_mask, 
                tgt_key_padding_mask=tgt_key_padding_mask
            ) # chuyển dữ liệu vào decoder và nhận output

            # mới  đổi cách tính loss
            logits = self.textual.generator(output.last_hidden_state)
            logits = torch.flatten(logits, start_dim=0, end_dim=1)
            tgt_output = torch.flatten(tgt_output) # chuyển sang 1 chiều
            loss = self.loss(logits, tgt_output.to(self.device))

            #end mới 
            output_dict: Dict[str, Any] = {
                "loss": loss,
                # Single scalar per batch for logging in training script.
                "loss_components": {"captioning_forward": loss}, # loss.clone.detach là 1 tensor
            }
            # Do captioning in backward direction if specified.
            if self.caption_backward:
                backward_caption_tokens = batch["noitpac_tokens"]

                backward_tgt_input = backward_caption_tokens[:, :-1]
                backward_tgt_output = backward_caption_tokens[:, 1:]
                tgt_mask = build_mask(backward_tgt_input).to(self.device) # chuyển về ma trận true false (j>i)
                tgt_key_padding_mask = build_key_padding_mask(backward_tgt_input, self.padding_idx).to(self.device) # seq có bằng pad_idx không lưu vào ma trận                
                memory = self.backward_textual.encode(src=visual_features.to(self.device)) # đưa dữ liệu vào encode và nhận được memory là ma trận 

                output = self.backward_textual.decode(
                    tgt=backward_tgt_input.to(self.device), 
                    memory=memory, 
                    tgt_mask=tgt_mask, 
                    tgt_key_padding_mask=tgt_key_padding_mask
                ) # chuyển dữ liệu vào decoder và nhận output      

                logits = self.backward_textual.generator(output.last_hidden_state)
                logits = torch.flatten(logits, start_dim=0, end_dim=1)
                backward_tgt_output = torch.flatten(backward_tgt_output) # chuyển sang 1 chiều
                backward_loss = self.loss(logits, backward_tgt_output.to(self.device))
                output_dict["loss"] += backward_loss

                # Single scalar per batch for logging in training script.
                output_dict["loss_components"].update(
                    captioning_backward=backward_loss
                )
                # end mới

            if not self.training: # cái nào thêm vô 
                # During validation (while pretraining), get best prediction
                # at every timestep.
                output_dict["predictions"] = [torch.argmax(x, dim=-1) for x in logits]
        else:
            if self.decoder is None:
                raise ValueError("Decoder for predicting captions is missing!")
            response_tracker = self.decoder.search(visual_features,self.textual)
            output_dict = {"predictions": response_tracker } 
        return output_dict


    def decoding_step(
        self, visual_features: torch.Tensor, partial_captions: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Given visual features and a batch of (giả định) partial captions, predict
        the logits thông qua output vocabulary tokens cho next timestep. phương thức này
        được sử dụng bởi :class:`~virtex.utils.beam_search.AutoRegressiveBeamSearch`
        và :class:`~virtex.utils.nucleus_sampling.AutoRegressiveNucleusSampling`.

        .. note::

            For nucleus sampling, ``beam_size`` sẽ luôn là 1 (không liên quan).

        Args:
            projected_visual_features: A tensor of shape ``(batch_size, ...,
                textual_feature_size)`` with visual features already projected to
                ``textual_feature_size``.
            partial_captions: A tensor of shape ``(batch_size * beam_size, timesteps)``
                containing tokens predicted so far -- one for each beam. We need all
                prior predictions because our model is auto-regressive.

        Returns:
            A tensor of shape ``(batch_size * beam_size, vocab_size)`` -- logits
            over output vocabulary tokens for next timestep.
        """

        # Expand and repeat image features while doing beam search.
        batch_size, channels, height, width = visual_features.size() # batch size = 1 , chanel = 2048, height = 7, width=7
        beam_size = int(partial_captions.size(0) / batch_size)
        if beam_size > 1:
            # shape: (batch_size * beam_size, channels, height, width)
            visual_features = visual_features.unsqueeze(1).repeat(1, beam_size, 1, 1, 1)
            visual_features = visual_features.view(
                batch_size * beam_size, channels, height, width
            )

        # Provide caption lengths as current length (irrespective of predicted
        # EOS/padding tokens). shape: (batch_size, )
        caption_lengths = torch.ones_like(partial_captions)
        if len(caption_lengths.size()) == 2:
            caption_lengths = caption_lengths.sum(1)
        else:
            # Add a timestep. shape: (batch_size, 1)
            partial_captions = partial_captions.unsqueeze(1)

        # shape: (batch_size * beam_size, partial_caption_length, vocab_size)
        logits = self.textual(visual_features, partial_captions, caption_lengths)
        # Return logits from the last timestep.
        return logits[:, -1, :]

    def log_predictions(
        self, batch: Dict[str, torch.Tensor], tokenizer: SentencePieceBPETokenizer
    ) -> str:

        self.eval()
        with torch.no_grad():
            predictions = self.forward(batch)["predictions"]
        self.train()

        predictions_str = ""
        for tokens, preds in zip(batch["caption_tokens"], predictions):
            predictions_str += f"""
                Caption tokens : {" ".join(tokens.tolist())}
                Predictions (f): {" ".join(preds.tolist())}

                """
        return predictions_str


class ForwardCaptioningModel(CaptioningModel):
    r"""
    Convenient extension of :class:`~virtex.models.captioning.CaptioningModel`
    for better readability: this passes ``caption_backward=False`` to super class.
    """

    def __init__(
        self,
        visual: VisualBackbone,
        textual: TextualHead,
        sos_index: int = 1,
        eos_index: int = 2,
        decoder: Any = None,
    ):
        super().__init__(
            visual,
            textual,
            sos_index=sos_index,
            eos_index=eos_index,
            caption_backward=False,
            decoder=decoder,
        )


class BidirectionalCaptioningModel(CaptioningModel):
    r"""
    Convenient extension of :class:`~virtex.models.captioning.CaptioningModel`
    for better readability: this passes ``caption_backward=True`` to super class.
    """

    def __init__(
        self,
        visual: VisualBackbone,
        textual: TextualHead,
        sos_index: int = 1,
        eos_index: int = 2,
        decoder: Any = None,
    ):
        super().__init__(
            visual,
            textual,
            sos_index=sos_index,
            eos_index=eos_index,
            caption_backward=True,
            decoder=decoder,
        )


# Convenient handle for our main model.
VirTexModel = BidirectionalCaptioningModel

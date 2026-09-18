"""Composes the submodules into one model for checkpointing and optimizer grouping."""
import itertools

import torch.nn as nn

from models.adversary import Adversary
from models.decoder import ActionDecoder
from models.h_encoder import HEncoder
from models.state_encoder import StateEncoder
from models.tokenizer import CanonicalTokenizer
from models.vision import VisionEncoder


class TactilePolicy(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.tokenizer = CanonicalTokenizer(num_links=cfg["num_links"], W=cfg["W"], d_tok=cfg["d_tok"])
        self.vision = VisionEncoder(
            d_model=cfg["d_model"], num_queries=cfg.get("num_vision_queries", 16),
            view_dropout_p=cfg.get("view_dropout_p", 0.1),
        )
        self.h_encoder = HEncoder(d_tok=cfg["d_tok"], d_h=cfg["d_h"], h_T_dim=cfg["h_T_dim"])
        self.state_encoder = StateEncoder(
            max_dof=cfg["max_dof"], d_tok=cfg["d_tok"], d_model=cfg["d_model"], d_h=cfg["d_h"],
            num_views=cfg["num_views"], num_vision_queries=cfg.get("num_vision_queries", 16),
        )
        self.decoder = ActionDecoder(embodiments=cfg["embodiments"], H=cfg["H"], d_model=cfg["d_model"])
        self.adversary = Adversary(d_model=cfg["d_model"], h_T_dim=cfg["h_T_dim"])

    def encoder_parameters(self):
        modules = [self.tokenizer, self.vision.proj, self.vision.pool, self.h_encoder, self.state_encoder]
        return (p for p in itertools.chain(*(m.parameters() for m in modules)) if p.requires_grad)

    def decoder_parameters(self):
        return (p for p in self.decoder.parameters() if p.requires_grad)

    def adversary_parameters(self):
        return (p for p in self.adversary.parameters() if p.requires_grad)

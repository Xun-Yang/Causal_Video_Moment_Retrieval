from dataclasses import dataclass
import torch

# temporal localization grounding 
@dataclass
class TLGBatch(object):
    # frames: list # [ImageList]
    feats: torch.tensor 
    feats_neg: torch.tensor
    queries: torch.tensor
    wordlens: torch.tensor
    
    def to(self, device):
        # self.frames = [f.to(device) for f in self.frames]
        self.feats = self.feats.to("cuda")
        self.feats_neg = self.feats_neg.to("cuda")
        self.queries = self.queries.to("cuda")
        self.wordlens = self.wordlens.to("cuda")
        
        return self
    


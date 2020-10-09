import torch
import torch.nn as nn

class attention(nn.Module):
    def __init__(self, embed_size, head):
        super(attention, self).__init__()
        self.embed_size = embed_size
        self.head = head
        self.head_dimension = embed_size // head

        assert (self.head_dimension * head == embed_size), #embed_size needs to be div by head

        self.values = nn.Linear
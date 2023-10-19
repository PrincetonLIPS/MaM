from .layers import get_timestep_embedding, default_init, ddpm_conv3x3
import torch.nn as nn
import torch

def image_float_to_int(x, binary=False):
    if binary:
        return torch.round( (x+1.0)/2.0 ).long() # from {-1.0, 1.0} to {0, 1}
    else:
        return torch.round( (x+1) * 127.5 ).long()

class InputProcessingImage(nn.Module):
    """Inputs embedding for images."""
    def __init__(self, num_classes, num_channels, input_channels, max_time):
        super().__init__()

        self.num_classes = num_classes
        self.num_channels = num_channels
        self.max_time = max_time
        self.act = nn.SiLU()
        self.input_channels = input_channels # 3 for RGB

        # timestep embedding
        self.M0 = nn.Linear(self.num_channels, self.num_channels * 4)
        self.M0.weight.data = default_init()(self.M0.weight.data.shape)
        nn.init.zeros_(self.M0.bias)
        self.M1 = nn.Linear(self.num_channels * 4, self.num_channels * 4)
        self.M1.weight.data = default_init()(self.M1.weight.data.shape)
        nn.init.zeros_(self.M1.bias)

        assert(self.num_channels % 4 == 0)
        self.M2 = ddpm_conv3x3(self.input_channels * 2, self.num_channels * 3 // 4)

        # timestep embedding
        self.M3 = nn.Embedding(self.num_classes, self.num_channels // 4)
        self.M4 = nn.Linear(self.input_channels * self.num_channels // 4, self.num_channels // 4)
        self.M4.weight.data = default_init()(self.M4.weight.data.shape)
        nn.init.zeros_(self.M4.bias)



    def forward(self, x, t, mask):
        assert(self.num_classes >= 1)
        assert(x.dtype == torch.float32)

        xint = image_float_to_int(x)
        x = torch.cat([x, mask], dim=1)

        # Timestep embedding
        temb = get_timestep_embedding(t, self.num_channels, self.max_time)
        temb = self.M0(temb)
        temb = self.M1(self.act(temb))
        assert(temb.shape == (t.shape[0], self.num_channels * 4))

        # Assign 3/4 of channels to the standard 'float' representation of the
        # inputs.
        h_first = self.M2(x)

        # # Here a 4th of the channels will be dedicated to the class embeddings.
        xint_permute = xint.permute(0, 2, 3, 1)
        emb_x = self.M3(xint_permute)
        emb_x = emb_x.reshape( *xint_permute.shape[:-1], self.input_channels * self.num_channels // 4 )
        
        h_emb_x = self.M4(emb_x)
        h_emb_x = h_emb_x.permute(0, 3, 1, 2)

        h_first = torch.cat( [ h_first, h_emb_x ], dim=1)

        return h_first, temb


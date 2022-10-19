import torch
import torch.nn as nn
from tqdm import tqdm as tqdm

def get_act(act_type):
    """Returns an activation function of type [act_type]."""
    if act_type == "gelu":
        return nn.GELU()
    else:
        raise NotImplementedError(f"Unknown activation '{act_type}'")

class MLP(nn.Module):
    def __init__(self, in_dim=None, h_dim=256, out_dim=42, layers=4, act_type="gelu", 
        end_with_act=True):
        super(MLP, self).__init__()

        if layers == 1 and end_with_act:
            self.model =  nn.Sequential(nn.Linear(in_dim, out_dim), get_act(act_type))
        elif layers == 1 and not end_with_act:
            self.model = nn.Linear(in_dim, out_dim)
        elif layers > 1:
            layer1 =  nn.LazyLinear(h_dim) if in_dim is None else nn.Linear(in_dim, h_dim)
            middle_layers = [nn.Linear(h_dim, h_dim) for _ in range(layers - 2)]
            layerN = nn.Linear(h_dim, out_dim)
            linear_layers = [layer1] + middle_layers + [layerN]

            layers = []
            for idx,l in enumerate(linear_layers):
                layers.append(l)
                if end_with_act:
                    layers.append(get_act(act_type))
                elif not end_with_act and idx < len(linear_layers) - 1:
                    layers.append(get_act(act_type))
                else:
                    continue
            
            self.model = nn.Sequential(*layers)
        
    def forward(self, x): return self.model(x)

class AdaIN(nn.Module):
    """AdaIN adapted for a transformer. Expects a BSxNPxC batch of images, where
    each image is represented as a set of P tokens, and BSxPxZ noise. This noise
    is mapped to be BSxPxC. The standard deviation and mean are taken over the
    first dimension, giving a BSxC tensors giving a mean and standard deviation.
    These are used to scale the image patches, ie. in the ith image, the kth
    element of the jth patch is scaled identically to the kth element of any
    other patch in that image.
    """
    def __init__(self, c, epsilon=1e-6, act_type="gelu"):
        super(AdaIN, self).__init__()
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.mapping_net = MLP(in_dim=512,
            h_dim=512,
            layers=8,
            out_dim=c,
            act_type=act_type)

    def get_latent_spec(self, x): return (x.shape[1], 512)

    def forward(self, x, z):
        """
        Args:
        x   -- image features
        z   -- latent codes
        """
        z = self.mapping_net(z) # BSxPxC
        x = torch.repeat_interleave(x, z.shape[0] // x.shape[0], dim=0)
        x_mean = torch.mean(x, keepdim=True, dim=-1)
        x_std = torch.std(x, keepdim=True, dim=-1) + self.epsilon
        z_mean = torch.mean(z, keepdim=True, dim=-1)
        z_std = torch.std(z, keepdim=True, dim=-1) + self.epsilon

        x_normalized = (x - x_mean.expand(*x.shape)) / x_std.expand(*x.shape)
        return z_std.expand(*x.shape) * x_normalized + z_mean.expand(*x.shape)


class VariationalBottleneck(nn.Module):

    def __init__(self, encoder_layers=2, encoder_h_dim=2048, h_dim=128, 
        decoder_layers=2, decoder_h_dim=2048, normalize_z=True,
        decoder_out_dim=768, act_type="gelu"):
        super(VariationalBottleneck, self).__init__()
        self.normalize_z = normalize_z
        self.encoder = MLP(in_dim=decoder_out_dim,
            h_dim=encoder_h_dim,
            out_dim=h_dim,
            layers=encoder_layers,
            act_type=act_type)

        self.decoder = MLP(in_dim=h_dim,
            h_dim=decoder_h_dim,
            out_dim=decoder_out_dim,
            layers=decoder_layers,
            act_type=act_type,
            end_with_act=False)

    def get_latent_spec(self, x):
        return self.encoder(x).shape[1:]
    
    def forward(self, x, z):
        z = nn.functional.normalize(z, dim=-1) if self.normalize_z else z
        fx = self.encoder(x)
        fx = torch.repeat_interleave(fx, z.shape[0] // fx.shape[0], dim=0)
        fx = fx + z
        fx = self.decoder(fx)
        return fx




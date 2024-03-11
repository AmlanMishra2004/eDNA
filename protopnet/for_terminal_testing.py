# Copy and paste this into the terminal
# This tests ppnet.cosine_similarity()

import torch 
import sys
sys.path.append('..')
import models   
import ppnet as ppn

model_path = "../small_best_updated_backbone.pt"
backbone = models.Small_Best_Updated()
backbone.load_state_dict(torch.load(model_path))
backbone.linear_layer = torch.nn.Identity() # remove the linear layer

ptype_shape = (156*2, 512, 25)

ppnet = ppn.construct_PPNet(
    features=backbone,
    pretrained=True,
    sequence_length=70,
    prototype_shape=ptype_shape,
    num_classes=156,
    prototype_activation_function='log',
    latent_weight=0.8,
)

x = torch.randn(64, 512, 30)
ptypes = torch.randn(ptype_shape)
ppnet.cosine_similarity(x, ptypes)
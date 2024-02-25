# previously model.py

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

# from resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
# from densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
# from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
#                          vgg19_features, vgg19_bn_features
# from base_model import base_model_features # COMMENTED OUT
# from receptive_field import compute_proto_layer_rf_info_v2

# base_architecture_to_features = {'base_feature': base_model_features} # COMMENTED OUT

#                                   'resnet18': resnet18_features,
#                                  'resnet34': resnet34_features,
#                                  'resnet50': resnet50_features,
#                                  'resnet101': resnet101_features,
#                                  'resnet152': resnet152_features,
#                                  'densenet121': densenet121_features,
#                                  'densenet161': densenet161_features,
#                                  'densenet169': densenet169_features,
#                                  'densenet201': densenet201_features,
#                                  'vgg11': vgg11_features,
#                                  'vgg11_bn': vgg11_bn_features,
#                                  'vgg13': vgg13_features,
#                                  'vgg13_bn': vgg13_bn_features,
#                                  'vgg16': vgg16_features,
#                                  'vgg16_bn': vgg16_bn_features,
#                                  'vgg19': vgg19_features,
#                                  'vgg19_bn': vgg19_bn_features}

class PPNet(nn.Module):

    # latent_weight controls how much to weight the latent
    #   space as opposed to our concatenated input
    # prototype_activation_function could be 'log', 'linear',
    #   or a generic function that converts distance to similarity score
    # self.features has to be named features to allow the precise loading
    def __init__(self, features, sequence_length, prototype_shape,
                #  proto_layer_rf_info,
                 num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck',
                 latent_weight=0.8):

        super(PPNet, self).__init__()
        self.sequence_length = sequence_length
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4
        assert latent_weight >= 0 and latent_weight <= 1, \
            "Error: Latent weight must be in [0, 1]"
        self.latent_weight = latent_weight
        self.prototype_activation_function = prototype_activation_function
        '''
        Here we are initializing the class identities of the prototypes.
        Without domain specific knowledge we allocate the same number of
        prototypes for each class.
        '''
        assert(self.num_prototypes % self.num_classes == 0)
        # a one-hot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(
            self.num_prototypes,
            self.num_classes
        )
        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1
        # self.proto_layer_rf_info = proto_layer_rf_info
        self.features = features

        # features_name = str(self.features).upper()
        # if features_name.startswith('VGG') or features_name.startswith('RES'):
        #     first_add_on_layer_in_channels = \
        #         [i for i in features.modules() if isinstance(i, nn.Conv1d)][-1].out_channels
        # elif features_name.startswith('DENSE'):
        #     first_add_on_layer_in_channels = \
        #         [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        # else:
        #     raise Exception('other base base_architecture NOT implemented')

        first_add_on_layer_in_channels = [i for i in features.modules() if \
                                    isinstance(i, nn.Conv1d)][-1].out_channels
        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv1d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv1d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        elif add_on_layers_type == 'identity':
            self.add_on_layers = nn.Sequential(nn.Identity())
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv1d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x = self.add_on_layers(x)
        return x

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * l
        filter of shape P * c * l1
        weight of shape P * c * l1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv1d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv1d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv1d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1)

        xp = F.conv1d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances
    
    # normalizes along each channel then returns the summed dot product
    # for each prototype
    def cosine_similarity(self, x, prototypes):
        # Shape of X: torch.Size([64, 512, 30])
        # 64 examples (in one batch), each with 512 channels and 30 sequence length
        # Shape of prototypes: torch.Size([1560, 512, 5])
        # 1560 prototypes (10 for each class), each with 512 channels and 5 sequence length
        # print(f"In cosine_similarity()...")
        # print(f"\tShape of X: {x.shape}")
        # print(f"\tShape of prototypes: {prototypes.shape}")
        # Normalize for each position in the sequence.
        # x_normalized will have the same shape as x, but each 30-element
        # vector along the last dimension will be a unit vector. This means
        # that the Euclidean norm (or length) of each of these vectors will
        # be 1. The same for p_normalized.

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_normalized = x / (self.epsilon + x_norm)

        p_norm = torch.norm(prototypes, p=2, dim=1, keepdim=True)
        p_normalized = prototypes / (self.epsilon + p_norm)

        # print(f"\tShape of x_normalized: {x_normalized.shape}")
        # print(f"\tShape of p_normalized: {p_normalized.shape}")
        
        similarities = F.conv1d(input=x_normalized, weight=p_normalized)
        return similarities

    def prototype_distances(self, x):
        '''
        x is the raw input
        should be (batch=64, channels=4, width=60,64,71) or (64, 4, 1, 71)
        '''
        # print(f"In prototype_distances, ")
        # print(f"\tShape of raw input x in prototype_distances() : {x.shape}")
        # Run the input through the convolutional layers.
        conv_features = self.conv_features(x)
        # avg_pooled_x = F.avg_pool1d(x, 2, stride=2)
        # CHANGED: concatenate (stack, below) instead of average (above)
        X1_inds = [a*2 for a in range(int(x.shape[-1]/2))] # adjust 15 (?)
        X2_inds = [a+1 for a in X1_inds]
        x_stacked = torch.concat([
            x[:, :, X1_inds],
            x[:, :, X2_inds]
        ], dim=1)
        # print(f"\t\tShape of conv_features: {conv_features.shape}")
        # print(f"\t\tShape of self.prototype_vectors: {self.prototype_vectors.shape}")
        # print(f"\t\tShape of self.prototype_vectors[:, :-8]: {self.prototype_vectors[:, :-8].shape}")
        # print(f"\t\tShape of x_stacked: {x_stacked.shape}")
        # print(f"\t\tShape of self.prototype_vectors: {self.prototype_vectors.shape}")
        # print(f"\t\tShape of self.prototype_vectors[:, -8:]: {self.prototype_vectors[:, -8:].shape}")
        latent_distances = self.cosine_similarity(conv_features, self.prototype_vectors[:, :-8])
        input_distances = self.cosine_similarity(x_stacked, self.prototype_vectors[:, -8:])
        # latent_distances = self.cosine_similarity(conv_features, self.prototype_vectors)
        # input_distances = self.cosine_similarity(avg_pooled_x, self.prototype_vectors[:, -4:])
        return self.latent_weight * latent_distances + (1 - self.latent_weight) * input_distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        # elif self.prototype_activation_function == 'inverse':
        #     return 1.0 / (distances + self.epsilon)
        # elif self.prototype_activation_function == 'exponential':
        #     return torch.exp(-distances)
        # elif self.prototype_activation_function == 'squared_inverse':
        #     return 1.0 / torch.pow(distances + self.epsilon, 2)
        else:
            return self.prototype_activation_function(distances)
        
    def forward(self, x):
        # print("Starting forward()!")
        distances = self.prototype_distances(x)
        '''
        we cannot refactor the lines below for similarity scores
        because we need to return max_similarities
        '''
        # global min pooling
        max_similarities = F.max_pool1d(distances,
                                      kernel_size=(distances.size()[2]))
        max_similarities = max_similarities.view(-1, self.num_prototypes)
        # prototype_similarities = self.distance_2_similarity(max_similarities)
        logits = self.last_layer(max_similarities)
        return logits, max_similarities

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output = self.conv_features(x)
        # avg_pooled_x = F.avg_pool1d(x, 2, stride=2)
        X1_inds = [a*2 for a in range(int(x.shape[-1]/2))] # adjust 15 (?)
        X2_inds = [a+1 for a in X1_inds]
        x_stacked = torch.concat([
            x[:, :, X1_inds],
            x[:, :, X2_inds]
        ], dim=1)
        concat = torch.cat((conv_output, x_stacked), dim=-2)
        # concat = torch.cat((conv_output, avg_pooled_x), dim=-2)
        # print(f"In push_forward: Shape of concat: {concat.shape}")
        # print(f"In push_forward: Shape of self.prototype_vectors: {self.prototype_vectors.shape}")
        similarities = self.cosine_similarity(concat, self.prototype_vectors)
        return concat, similarities

    def prune_prototypes(self, prototypes_to_prune):
        '''
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...],
                                              requires_grad=True)

        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...],
                                 requires_grad=False)
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def __repr__(self):
        # PPNet(self, features, sequence_length, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\tsequence_length: {},\n'
            '\tprototype_shape: {},\n'
            # '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.sequence_length,
                          self.prototype_shape,
                        #   self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv1d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

# features: backbone with loaded weights
# prototype_shape = (num prototypes, input channel, spatial dim, spatial dim)
def construct_PPNet(features, pretrained=True, sequence_length=60,
                    prototype_shape=(2*156, 512, 1), num_classes=156,
                    prototype_activation_function='log',
                    add_on_layers_type='identity', latent_weight=0.8):
    # features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    # layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    # proto_layer_rf_info = compute_proto_layer_rf_info_v2(sequence_length=sequence_length,
    #                                                      layer_filter_sizes=layer_filter_sizes,
    #                                                      layer_strides=layer_strides,
    #                                                      layer_paddings=layer_paddings,
    #                                                      prototype_kernel_size=prototype_shape[2])
    return PPNet(features=features,
                 sequence_length=sequence_length,
                 prototype_shape=prototype_shape,
                #  proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type, 
                 latent_weight=latent_weight)
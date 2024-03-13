# previously model.py

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

class PPNet(nn.Module):

    # latent_weight controls how much to weight the latent
    #   space as opposed to our concatenated input
    # prototype_activation_function could be 'log', 'linear',
    #   or a generic function that converts distance to similarity score
    # self.features has to be named features to allow the precise loading
    def __init__(self, features, sequence_length, prototype_shape,
                #  proto_layer_rf_info,
                 num_classes,
                 prototype_activation_function='log',
                 latent_weight=0.8):

        super(PPNet, self).__init__()
        self.sequence_length = sequence_length
        self.prototype_shape = prototype_shape
        assert prototype_shape[2] % 2 != 0, \
            "Error: Prototype length must be odd, since it needs a center."
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4
        assert latent_weight >= 0 and latent_weight <= 1, \
            "Error: Latent weight must be in [0, 1]"
        
        self.latent_weight = latent_weight
        self.prototype_activation_function = prototype_activation_function

        # Here we are initializing the class identities of the prototypes.
        # Without domain specific knowledge we allocate the same number of
        # prototypes for each class.

        assert(self.num_prototypes % self.num_classes == 0)
        # a one-hot indication matrix for each prototype's class identity, 
        # sam: ex. [512, 156], tells you how well each prototype matches to each class?
        self.prototype_class_identity = torch.zeros(
            self.num_prototypes,
            self.num_classes
        )
        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1
        # tensor([[1., 0., 0.,  ..., 0., 0., 0.],
        #         [1., 0., 0.,  ..., 0., 0., 0.],
        #         [0., 1., 0.,  ..., 0., 0., 0.],
        #         ...,
        #         [0., 0., 0.,  ..., 0., 1., 0.],
        #         [0., 0., 0.,  ..., 0., 0., 1.],
        #         [0., 0., 0.,  ..., 0., 0., 1.]])
            
        # self.proto_layer_rf_info = proto_layer_rf_info
        self.features = features

        # initializes all prototype values to [0, 1)
        # prototype_shape is: ex. [156*2, 512+8, 25]
        # [config['num_classes']*ptypes_per_class, num_latent_channels+8, ptype_length]
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        return x
    
    # normalizes along each channel then returns the summed dot product
    # for each prototype
    def cosine_similarity(self, x, prototypes):
        # Shape of X: without -8: torch.Size([64, 512, 30])(old) with: ([94, 512, 35])
        # 64 examples (in one batch), each with 512 channels and 30 sequence length
        # Shape of prototypes: torch.Size([1560, 512, 5])
        # 1560 prototypes (10 for each class), each with 512 channels and 5 sequence length
        # Normalize for each position in the sequence.
        # x_normalized will have the same shape as x, but each 30-element
        # vector along the last dimension will be a unit vector. This means
        # that the Euclidean norm (or length) of each of these vectors will
        # be 1. The same for p_normalized.

        # Compute L2 (euclidean) norm along each channel. (sqrt(sum(x**2 for each x)))
        # Divide each element by its norm (plus self.epsilon for numerical stability)
        # Divide each element by sqrt(prototype length)

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_normalized = x / (self.epsilon + x_norm)
        x_normalized /= float(prototypes.shape[-1])**0.5

        p_norm = torch.norm(prototypes, p=2, dim=1, keepdim=True)
        p_normalized = prototypes / (self.epsilon + p_norm)
        p_normalized /= float(prototypes.shape[-1])**0.5
    
        similarities = F.conv1d(input=x_normalized, weight=p_normalized)
        # print(f"\nIn cosine_similarity()...")
        # print(f"\tShape of X: {x.shape}") # without -8: ([94, 512, 35]), with: ([94, 512, 35])
        # print(f"\tShape of prototypes: {prototypes.shape}") # without -8: ([312, 512, 25]), with: ([312, 512, 25])  
        # print(f"\tShape of x_normalized: {x_normalized.shape}") # ([94, 512, 35])
        # print(f"\tShape of p_normalized: {p_normalized.shape}") # ([312, 512, 25])
        # print(f"\tShape of similarities: {similarities.shape}") # without -8: ([94, 312, 11]), with: ([94, 312, 11])
        # wait = input("PAUSE")
        return similarities
    
    # def cosine_similarity(self, x, prototypes):
    #     # Compute the L2 norm of x and prototypes along each channel
    #     x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
    #     p_norm = torch.norm(prototypes, p=2, dim=1, keepdim=True)

    #     # Normalize x and prototypes by their L2 norms
    #     x_normalized = x / (self.epsilon + x_norm)
    #     p_normalized = prototypes / (self.epsilon + p_norm)

    #     # Compute the dot product of x_normalized and p_normalized
    #     dot_product = torch.sum(x_normalized * p_normalized, dim=1)

    #     # Compute the cosine similarity
    #     cosine_similarity = dot_product / (torch.norm(x_normalized, p=2, dim=1) * torch.norm(p_normalized, p=2, dim=1) + self.epsilon)

    #     return cosine_similarity

    def prototype_distances(self, x):
        '''
        x is the raw input, (batch=64, channels=4, width=70)
        '''
        # Run the input through the convolutional layers -> (batch=156, channels=512, width=30)
        conv_features = self.conv_features(x)
        # avg_pooled_x = F.avg_pool1d(x, kernel_size=2, stride=2)
        # CHANGED: concatenate (stack, below) instead of average (above). Doubles the # of channels.
        even_indexes = [a*2 for a in range(int(x.shape[-1]/2))]
        odd_indexes = [a+1 for a in even_indexes]
        x_stacked = torch.concat([
            x[:, :, even_indexes],
            x[:, :, odd_indexes]
        ], dim=1)

        # The average method halved the length and kept the number of channels
        # Concat halves the length of the raw sequences and doubles the number of channels
        # If x were a single batch:
        # [[[0, 1, 2, 3,..., 69],
        #   [0, 1, 2, 3,..., 69],
        #   [0, 1, 2, 3,..., 69],
        #   [0, 1, 2, 3,..., 69]]]
        # x_stacked would be:
        # [[[0, 2, 4,..., 68], # even indexes
        #   [0, 2, 4,..., 68],
        #   [0, 2, 4,..., 68],
        #   [0, 2, 4,..., 68],
        #   [1, 3, 5,..., 69], # odd indexes
        #   [1, 3, 5,..., 69],
        #   [1, 3, 5,..., 69],
        #   [1, 3, 5,..., 69]]]

        latent_distances = self.cosine_similarity(conv_features, self.prototype_vectors[:, :-8, :])
        input_distances = self.cosine_similarity(x_stacked, self.prototype_vectors[:, -8:, :])
        print(f"latent_distances: {latent_distances}")
        print(f"input_distances: {input_distances}")
        # latent_distances = self.cosine_similarity(conv_features, self.prototype_vectors)
        # input_distances = self.cosine_similarity(avg_pooled_x, self.prototype_vectors[:, -4:])
        return self.latent_weight * latent_distances + (1 - self.latent_weight) * input_distances
    
        # print(f"In prototype_distances, ")
        # print(f"\tShape of raw input x in prototype_distances() : {x.shape}")
        # print(f"\t\tShape of conv_features: {conv_features.shape}")
        # print(f"\t\tShape of self.prototype_vectors: {self.prototype_vectors.shape}")
        # print(f"\t\tShape of self.prototype_vectors[:, :-8]: {self.prototype_vectors[:, :-8].shape}")
        # print(f"\t\tShape of x_stacked: {x_stacked.shape}")
        # print(f"\t\tShape of self.prototype_vectors: {self.prototype_vectors.shape}")
        # print(f"\t\tShape of self.prototype_vectors[:, -8:]: {self.prototype_vectors[:, -8:].shape}")
        
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
        # print(f"Max similarities in forward(): {max_similarities}")
        return logits, max_similarities

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output = self.conv_features(x)
        # avg_pooled_x = F.avg_pool1d(x, 2, stride=2)
        # 
        even_indexes = [a*2 for a in range(int(x.shape[-1]/2))] # adjust 15 (?)
        odd_indexes = [a+1 for a in even_indexes]
        x_stacked = torch.concat([
            x[:, :, even_indexes],
            x[:, :, odd_indexes]
        ], dim=1)
        concat = torch.cat((conv_output, x_stacked), dim=-2)
        # concat = torch.cat((conv_output, avg_pooled_x), dim=-2)
        # print(f"In push_forward: Shape of concat: {concat.shape}")
        # print(f"In push_forward: Shape of self.prototype_vectors: {self.prototype_vectors.shape}")
        # similarities = self.cosine_similarity(concat, self.prototype_vectors)
        distances = self.prototype_distances(x)
        max_similarities = F.max_pool1d(distances,
                                      kernel_size=(distances.size()[2]))
        max_similarities = max_similarities.view(-1, self.num_prototypes)
        # print(f"Max similarities in push_forward(): {max_similarities}")
        return concat, distances
        # return concat, similarities

    # Defines how you would print the model, ex. print(ppnet)
    def __repr__(self):
        # PPNet(self, features, sequence_length, prototype_shape,
        # proto_layer_rf_info, num_classes):
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

# features: backbone with loaded weights
# prototype_shape = (num prototypes, input channel, spatial dim, spatial dim)
def construct_PPNet(features, pretrained=True, sequence_length=60,
                    prototype_shape=(2*156, 512, 1), num_classes=156,
                    prototype_activation_function='log', latent_weight=0.8):
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
                 prototype_activation_function=prototype_activation_function,
                 latent_weight=latent_weight)






    # UNUSED
    # def distance_2_similarity(self, distances):
    #     if self.prototype_activation_function == 'log':
    #         return torch.log((distances + 1) / (distances + self.epsilon))
    #     elif self.prototype_activation_function == 'linear':
    #         return -distances
    #     # elif self.prototype_activation_function == 'inverse':
    #     #     return 1.0 / (distances + self.epsilon)
    #     # elif self.prototype_activation_function == 'exponential':
    #     #     return torch.exp(-distances)
    #     # elif self.prototype_activation_function == 'squared_inverse':
    #     #     return 1.0 / torch.pow(distances + self.epsilon, 2)
    #     else:
    #         return self.prototype_activation_function(distances)

    # UNUSED
    # @staticmethod
    # def _weighted_l2_convolution(input, filter, weights):
    #     '''
    #     input of shape N * c * l
    #     filter of shape P * c * l1
    #     weight of shape P * c * l1
    #     '''
    #     input2 = input ** 2
    #     input_patch_weighted_norm2 = F.conv1d(input=input2, weight=weights)

    #     filter2 = filter ** 2
    #     weighted_filter2 = filter2 * weights
    #     filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2))
    #     filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1)

    #     weighted_filter = filter * weights
    #     weighted_inner_product = F.conv1d(input=input, weight=weighted_filter)

    #     # use broadcast
    #     intermediate_result = \
    #         - 2 * weighted_inner_product + filter_weighted_norm2_reshape
    #     # x2_patch_sum and intermediate_result are of the same shape
    #     distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

    #     return distances

    # UNUSED
    # def _l2_convolution(self, x):
    #     '''
    #     apply self.prototype_vectors as l2-convolution filters on input x
    #     '''
    #     x2 = x ** 2
    #     x2_patch_sum = F.conv1d(input=x2, weight=self.ones)

    #     p2 = self.prototype_vectors ** 2
    #     p2 = torch.sum(p2, dim=(1, 2))
    #     # p2 is a vector of shape (num_prototypes,)
    #     # then we reshape it to (num_prototypes, 1, 1)
    #     p2_reshape = p2.view(-1, 1)

    #     xp = F.conv1d(input=x, weight=self.prototype_vectors)
    #     intermediate_result = - 2 * xp + p2_reshape  # use broadcast
    #     # x2_patch_sum and intermediate_result are of the same shape
    #     distances = F.relu(x2_patch_sum + intermediate_result)

    #     return distances

    # UNUSED
    # def set_last_layer_incorrect_connection(self, incorrect_strength):
    #     '''
    #     the incorrect strength will be actual strength if -0.5 then input -0.5
    #     '''
    #     positive_one_weights_locations = torch.t(self.prototype_class_identity)
    #     negative_one_weights_locations = 1 - positive_one_weights_locations

    #     correct_class_connection = 1
    #     incorrect_class_connection = incorrect_strength
    #     self.last_layer.weight.data.copy_(
    #         correct_class_connection * positive_one_weights_locations
    #         + incorrect_class_connection * negative_one_weights_locations)

    # UNUSED
    # def prune_prototypes(self, prototypes_to_prune):
    #     '''
    #     prototypes_to_prune: a list of indices each in
    #     [0, current number of prototypes - 1] that indicates the prototypes to
    #     be removed
    #     '''
    #     prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))

    #     self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...],
    #                                           requires_grad=True)

    #     self.prototype_shape = list(self.prototype_vectors.size())
    #     self.num_prototypes = self.prototype_shape[0]

    #     # changing self.last_layer in place
    #     # changing in_features and out_features make sure the numbers are consistent
    #     self.last_layer.in_features = self.num_prototypes
    #     self.last_layer.out_features = self.num_classes
    #     self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

    #     # self.ones is nn.Parameter
    #     self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...],
    #                              requires_grad=False)
    #     # self.prototype_class_identity is torch tensor
    #     # so it does not need .data access for value update
    #     self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]
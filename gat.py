# The based unit of graph convolutional networks.

import torch
import torch.nn as nn

class GAT(nn.Module):

    """
        Implementation #2 was inspired by the official GAT implementation: https://github.com/PetarV-/GAT

        It's conceptually simpler than implementation #3 but computationally much less efficient.

        Note: this is the naive implementation not the sparse one and it's only suitable for a transductive setting.
        It would be fairly easy to make it work in the inductive setting as well but the purpose of this layer
        is more educational since it's way less efficient than implementation 3.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.head_dim = 1
        
        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = 3
        self.num_out_features = out_channels
        self.concat = False  # whether we should concatenate or average the attention heads
        self.add_skip_connection = True
        self.dropout= nn.Dropout(p=0.1)

        num_in_features= in_channels  # consequence of concatenation
        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, self.num_of_heads * self.num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, self.num_of_heads, self.num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, self.num_of_heads, self.num_out_features))

        self.skip_proj = nn.Linear(num_in_features, self.num_out_features)
      
        #
        # End of trainable weights
        #

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.activation = False
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params()






        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        
    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

    
    def skip_concat_bias(self, in_nodes_features, out_nodes_features):

        # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
        # only imp1 will enter this one
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                #print('inside !')
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features
                #print('out_nodes_features (concat): ', out_nodes_features.shape)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features)
                #print('out_nodes_features: ', out_nodes_features.shape)
                n, v, c = out_nodes_features.shape
                out_nodes_features = out_nodes_features.view(n // 3, 3, v, c)
                #print('out_nodes_features: ', out_nodes_features.shape)

            
        # shape = (N, NH, FOUT) -> (N, FOUT)
        out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        return out_nodes_features

    
    def forward(self, x, A):

        #
        # Step 1: Linear Projection + regularization (using linear layer instead of matmul as in imp1)
        #

        #print('input: ', x.shape)
        batch, c_in, t, v = x.shape

        x = x.permute(0, 2, 3, 1)

        #print('input reshaped: ', x.shape)

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        in_nodes_features = self.dropout(x)

        #print('in_nodes_features (dropout): ', in_nodes_features.shape)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        #print('nodes_features_proj: ', nodes_features_proj.shape)

        nodes_features_proj_input = nodes_features_proj

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well
        #print('nodes_features_proj (dropout): ', nodes_features_proj.shape)

        #
        # Step 2: Edge attention calculation (using sum instead of bmm + additional permute calls - compared to imp1)
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1)
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = torch.sum((nodes_features_proj * self.scoring_fn_source), dim=-1, keepdim=True)
        scores_target = torch.sum((nodes_features_proj * self.scoring_fn_target), dim=-1, keepdim=True)
        #print('scores_source: ', scores_source.shape)
        #print('scores_target: ', scores_target.shape)

        # Reshape the tensor
        n, nh, c = scores_source.shape
        scores_source = scores_source.view(n // 21, 21, nh, c)
        scores_target = scores_target.view(n // 21, 21, nh, c)

        # src shape = (NH, N, 1) and trg shape = (NH, 1, N)
        scores_source = scores_source.transpose(1, 2)
        scores_target = scores_target.permute(0, 2, 3, 1)
        #print('scores_source: ', scores_source.shape)
        #print('scores_target: ', scores_target.shape)

        # shape = (NH, N, 1) + (NH, 1, N) -> (NH, N, N) with the magic of automatic broadcast <3
        # In Implementation 3 we are much smarter and don't have to calculate all NxN scores! (only E!)
        # Tip: it's conceptually easier to understand what happens here if you delete the NH dimension
        all_scores = self.leakyReLU(scores_source + scores_target)
        #print('all_scores: ', all_scores.shape)

        # Create a mask where the tensor equals 0
        mask = A == 0.0
        # Replace 0 values with -inf
        A[mask] = -float('inf')

        # connectivity mask will put -inf on all locations where there are no edges, after applying the softmax
        # this will result in attention scores being computed only for existing edges
        all_attention_coefficients = self.softmax(all_scores + A)
        #print('all_attention_coefficients: ', all_attention_coefficients.shape)

        #
        # Step 3: Neighborhood aggregation (same as in imp1)
        #
        n, nh, c = nodes_features_proj.shape
        nodes_features_proj = nodes_features_proj.view(n // 21, 21, nh, c).permute(0, 2, 1, 3)
        #print('nodes_features_proj: ', nodes_features_proj.shape)

        # Reshape both tensors to be 3D by combining the first two dimensions
        all_attention_coefficients = all_attention_coefficients.reshape(-1, 21, 21)  # Shape: [36800*3, 21, 21]
        nodes_features_proj = nodes_features_proj.reshape(-1, 21, c)  # Shape: [36800*3, 21, 256]
        
        

        # batch matrix multiply, shape = (NH, N, N) * (NH, N, FOUT) -> (NH, N, FOUT)
        out_nodes_features = torch.bmm(all_attention_coefficients, nodes_features_proj)
        #print('out_nodes_features: ', out_nodes_features.shape)
        b, n, c_in = out_nodes_features.shape
        out_nodes_features = out_nodes_features.view(b * n// 3 , 3, c_in)  # Shape: [5*7360*3, 21, 256]
        #print('out_nodes_features: ', out_nodes_features.shape)

        #
        # Step 4: Residual/skip connections, concat and bias (same as in imp1)
        #
        n, nh, c_out = nodes_features_proj_input.shape
        #nodes_features_proj_input = nodes_features_proj_input.reshape(-1, v, c_in)  # Shape: [5*7360, 21, 3]
        
        #print('nodes_features_proj_input: ', nodes_features_proj_input.shape)

        out_nodes_features = self.skip_concat_bias(nodes_features_proj_input, out_nodes_features)
        #print('out_nodes_features (skip connections): ', out_nodes_features.shape)

        out_nodes_features = out_nodes_features.view(batch , c_out, t, v)  # Shape: [5*7360*3, 21, 256]
        #print('out_nodes_features: ', out_nodes_features.shape)

        x = out_nodes_features

        return x.contiguous()

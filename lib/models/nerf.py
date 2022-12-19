import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import Embedding

class DeRF(nn.Module):
    def __init__(self,
                 D=6, W=128,
                 freqs_xyz=10,
                 deformation_dim=0,
                 out_channels=3,
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels: number of input channels for xyz (3+3*10*2=63 by default)
        skips: add skip connection in the Dth layer
        """
        super(DeRF, self).__init__()
        self.D = D
        self.W = W
        self.freqs_xyz = freqs_xyz
        self.deformation_dim = deformation_dim
        self.skips = skips

        self.in_channels = 3 + 3*freqs_xyz*2 + deformation_dim
        self.out_channels = out_channels

        self.encoding_xyz = Embedding(3, freqs_xyz)

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.in_channels, W)
            elif i in skips:
                layer = nn.Linear(W+self.in_channels, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)

        self.out = nn.Linear(W, self.out_channels)

    def forward(self, xyz, deformation_code=None):
        xyz_encoded = self.encoding_xyz(xyz)
        
        if self.deformation_dim > 0:
            xyz_encoded = torch.cat([xyz_encoded, deformation_code], -1)
        
        xyz_ = xyz_encoded
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([xyz_encoded, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
        out = self.out(xyz_)

        return out

class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 freqs_xyz=10, freqs_dir=4,
                 use_view=True, use_normal=False,
                 deformation_dim=0, appearance_dim=0,
                 skips=[4], actvn_type='relu'):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.freqs_xyz = freqs_xyz
        self.freqs_dir = freqs_dir
        self.deformation_dim = deformation_dim
        self.appearance_dim = appearance_dim
        self.skips = skips
        self.use_view = use_view
        self.use_normal = use_normal

        self.encoding_xyz = Embedding(3, freqs_xyz)
        if self.use_view:
            self.encoding_dir = Embedding(3, freqs_dir)

        self.in_channels_xyz = 3 + 3*freqs_xyz*2 + deformation_dim

        self.in_channels_dir = appearance_dim
        if self.use_view:
            self.in_channels_dir += 3 + 3*freqs_dir*2
        if self.use_normal:
            self.in_channels_dir += 3

        if actvn_type == 'relu':
            actvn = nn.ReLU(inplace=True)
        elif actvn_type == 'leaky_relu':
            actvn = nn.LeakyReLU(0.2, inplace=True)
        elif actvn_type == 'softplus':
            actvn = nn.Softplus(beta=100)
        else:
            assert NotImplementedError

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+self.in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, actvn)
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+self.in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

    def forward(self, xyz, viewdir=None, deformation_code=None, appearance_code=None):
        """
        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
        Outputs:
            out: (B, 4), rgb and sigma
        """
        sigma, xyz_encoding_final = self.get_sigma(xyz, deformation_code=deformation_code)

        dir_encoding_input = xyz_encoding_final

        if self.use_view:
            viewdir_encoded = self.encoding_dir(viewdir)
            dir_encoding_input = torch.cat([dir_encoding_input, viewdir_encoded], -1)
        if self.use_normal:
            normal = self.get_normal(xyz, deformation_code=deformation_code)
            dir_encoding_input = torch.cat([dir_encoding_input, normal], -1)
        if self.appearance_dim > 0:
            dir_encoding_input = torch.cat([dir_encoding_input, appearance_code], -1)

        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        return rgb, sigma

    def get_sigma(self, xyz, deformation_code=None, only_sigma=False):

        xyz_encoded = self.encoding_xyz(xyz)
        
        if self.deformation_dim > 0:
            xyz_encoded = torch.cat([xyz_encoded, deformation_code], -1)

        xyz_ = xyz_encoded
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([xyz_encoded, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)

        if only_sigma:
            return sigma
        
        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        return sigma, xyz_encoding_final

    def get_normal(self, xyz, deformation_code=None, delta=0.02):
        with torch.set_grad_enabled(True):
            xyz.requires_grad_(True)
            sigma = self.get_sigma(xyz, deformation_code=deformation_code, only_sigma=True)
            alpha = 1 - torch.exp(-delta * torch.relu(sigma))
            normal = torch.autograd.grad(
                outputs=alpha,
                inputs=xyz,
                grad_outputs=torch.ones_like(alpha, requires_grad=False, device=alpha.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

        return normal


####

class MLP(nn.Module):
    def __init__(self, 
                 filter_channels, 
                 merge_layer=0,
                 res_layers=[],
                 norm='group',
                 last_op=None):
        super(MLP, self).__init__()

        self.filters = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.merge_layer = merge_layer if merge_layer > 0 else len(filter_channels) // 2
        self.res_layers = res_layers
        self.norm = norm
        self.last_op = last_op

        for l in range(0, len(filter_channels)-1):
            if l in self.res_layers:
                self.filters.append(nn.Conv1d(
                    filter_channels[l] + filter_channels[0],
                    filter_channels[l+1],
                    1))
            else:
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l+1],
                    1))
            if l != len(filter_channels)-2:
                if norm == 'group':
                    self.norms.append(nn.GroupNorm(32, filter_channels[l+1]))
                elif norm == 'batch':
                    self.norms.append(nn.BatchNorm1d(filter_channels[l+1]))

        ## init
        # for l in range(0, len(filter_channels)-1):
        #     conv = self.filters[l]
        #     conv.weight.data.fill_(0.00001)
        #     conv.bias.data.fill_(0.0)

    def forward(self, feature):
        '''
        feature may include multiple view inputs
        args:
            feature: [B, C_in, N]
        return:
            [B, C_out, N] prediction
        '''
        y = feature
        tmpy = feature
        phi = None
        for i, f in enumerate(self.filters):
            y = f(
                y if i not in self.res_layers
                else torch.cat([y, tmpy], 1)
            )
            if i != len(self.filters)-1:
                if self.norm not in ['batch', 'group']:
                    y = F.leaky_relu(y)
                else:
                    y = F.leaky_relu(self.norms[i](y))         
            if i == self.merge_layer:
                phi = y.clone()

        if self.last_op is not None:
            y = self.last_op(y)

        return y #, phi


class GeoMLP(nn.Module):
    def __init__(self,
                 filter_channels = [128, 256, 128],
                 input_dim = 3, embedding_freqs = 10,
                 output_dim = 3,
                 cond_dim = 72,
                 last_op=torch.tanh,
                 scale=0.1):
        super(GeoMLP, self).__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.embedding_freqs = embedding_freqs
        self.embedding_dim = input_dim*(2*embedding_freqs+1)
        # Embeddings
        self.embedding = Embedding(self.input_dim, embedding_freqs) # 10 is the default number

        # xyz encoding layers
        filter_channels = [self.embedding_dim + cond_dim] + filter_channels + [output_dim]
        self.mlp = MLP(filter_channels, last_op=last_op)
        self.scale = scale
        
    def forward(self, x, cond):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py
        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)
        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        # x: [B, nv, 3]
        # cond: [B, n_theta]
        batch_size, nv, _ = x.shape
        pos_embedding = self.embedding(x.reshape(batch_size, -1)).reshape(batch_size, nv, -1)
        cond = cond[:,None,:].expand(-1, nv, -1)
        inputs = torch.cat([pos_embedding, cond], -1) #[B, nv, n_position+n_theta]
        inputs = inputs.permute(0,2,1)
        out = self.mlp(inputs).permute(0,2,1)*self.scale
        return out


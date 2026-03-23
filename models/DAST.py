import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from torch.nn.utils import weight_norm
import numpy as np


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.5):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultiScaleTCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_sizes=[2, 3, 5], dropout=0.5):
        super().__init__()
        self.tcns = nn.ModuleList([
            TemporalConvNet(num_inputs, num_channels, kernel_size=k, dropout=dropout)
            for k in kernel_sizes
        ])
        
        self.fusion_weights = nn.Parameter(torch.ones(len(kernel_sizes)))
        
    def forward(self, x):
        features = []
        for tcn in self.tcns:
            features.append(tcn(x))
        
        weights = F.softmax(self.fusion_weights, dim=0)
        weighted_features = [f * w for f, w in zip(features, weights)]
        
        output = torch.stack(weighted_features, dim=-1).sum(dim=-1)
        return output



class AdaptiveGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, node_n=39, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.node_n = node_n
        
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        
        self.static_adj = Parameter(torch.FloatTensor(node_n, node_n))
        self.dynamic_transform = nn.Sequential(
            nn.Linear(in_features, node_n * node_n // 4),
            nn.ReLU(),
            nn.Linear(node_n * node_n // 4, node_n * node_n)
        )
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.static_adj.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, x):
        if len(x.shape) == 3:
            B, V, C = x.shape
            T = 1
            x = x.unsqueeze(1)
        else:
            B, T, V, C = x.shape
            
        x_flat = x.view(B*T, V, C)
        dynamic_adj = self.dynamic_transform(x_flat.mean(dim=1))
        dynamic_adj = dynamic_adj.view(B*T, V, V)
        dynamic_adj = F.softmax(dynamic_adj, dim=-1)
        
        adj_matrix = self.static_adj.unsqueeze(0) + dynamic_adj
        
        support = torch.matmul(x_flat, self.weight)
        output = torch.matmul(adj_matrix, support)
        
        if self.bias is not None:
            output = output + self.bias
            
        output = output.view(B, T, V, -1)
        if T == 1:
            output = output.squeeze(1)
            
        return output



class HierarchicalGCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, node_n=39, num_layers=3, dropout=0.1):
        super().__init__()
        self.node_n = node_n
        self.input_feature = input_feature
        self.hidden_feature = hidden_feature

        self.input_adapter = nn.Sequential(
            nn.Linear(input_feature, hidden_feature),
            nn.LayerNorm(hidden_feature),
            nn.ReLU(),
            nn.Dropout(dropout)
        ) if input_feature != hidden_feature else nn.Identity()
        self.gcn_layers = nn.ModuleList()
        
        self.gcn_layers.append(
            nn.Sequential(
                AdaptiveGraphConvolution(input_feature, hidden_feature, node_n),
                nn.BatchNorm1d(node_n),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        )
        
        for _ in range(1, num_layers-1):
            self.gcn_layers.append(
                nn.Sequential(
                    AdaptiveGraphConvolution(hidden_feature, hidden_feature, node_n),
                    nn.BatchNorm1d(node_n),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        self.gcn_layers.append(
            nn.Sequential(
                AdaptiveGraphConvolution(hidden_feature, hidden_feature, node_n),
                nn.BatchNorm1d(node_n),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        )
        
        self.skip_connections = nn.ModuleList([
            nn.Linear(hidden_feature, hidden_feature) if i == 0 else nn.Identity()
            for i in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_feature * num_layers, hidden_feature)
        
    def forward(self, x):
        B, T, features = x.shape
        V = self.node_n
        if features != V * self.input_feature:
            x_flat = x.view(B*T, features)
            x_adapted = self.input_adapter(x_flat)
            C = self.hidden_feature
            x_reshaped = x_adapted.view(B*T, V, C)
        else:
            C = self.input_feature
            x_reshaped = x.view(B, T, V, C)
            x_reshaped = x_reshaped.view(B*T, V, C)
        
        all_features = []
        current = x_reshaped
        
        for i, (gcn_layer, skip) in enumerate(zip(self.gcn_layers, self.skip_connections)):
            gcn_out = gcn_layer(current)
            
            if i == 0:
                skip_input = current
                if isinstance(skip, nn.Linear):
                    skip_input = skip(skip_input)
                gcn_out = gcn_out + skip_input
            
            all_features.append(gcn_out)
            
            current = gcn_out
        
        combined = torch.cat(all_features, dim=-1)

        output = self.output_proj(combined)
        
        output = output.view(B, T, V * self.hidden_feature)
        
        return output



class SpatioTemporalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.spatial_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        B, T, V, C = x.shape

        x_spatial = x.view(B*T, V, C)
        spatial_attn_out, _ = self.spatial_attention(x_spatial, x_spatial, x_spatial)
        x_spatial = self.spatial_norm(x_spatial + spatial_attn_out)
        
        x_temporal = x_spatial.view(B, T, V, C).permute(0, 2, 1, 3).contiguous()
        x_temporal = x_temporal.view(B*V, T, C)
        temporal_attn_out, _ = self.temporal_attention(x_temporal, x_temporal, x_temporal)
        x_temporal = self.temporal_norm(x_temporal + temporal_attn_out)
        
        x_out = x_temporal.view(B, V, T, C).permute(0, 2, 1, 3).contiguous()
        
        x_out = x_out.view(B*T*V, C)
        ffn_out = self.ffn(x_out)
        x_out = self.ffn_norm(x_out + ffn_out)
        x_out = x_out.view(B, T, V, C)
        
        return x_out


class EnhancedSELayer(nn.Module):
    def __init__(self, channel, reduction=16, use_max_pooling=False):
        super().__init__()
        self.use_max_pooling = use_max_pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if use_max_pooling:
            self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        fc_input_dim = channel * (2 if use_max_pooling else 1)
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        original_shape = x.shape
        
        if len(original_shape) == 3:
            B, C, T = x.shape
            x = x.unsqueeze(-1)
            needs_squeeze = True
        elif len(original_shape) == 4:
            B, C, T, V = x.shape
            needs_squeeze = False
        
        avg_out = self.avg_pool(x).view(B, C)
        
        if self.use_max_pooling:
            max_out = self.max_pool(x).view(B, C)
            combined = torch.cat([avg_out, max_out], dim=1)
        else:
            combined = avg_out
            
        y = self.fc(combined).view(B, C, 1, 1)
        
        output = x * y.expand_as(x)
        
        if needs_squeeze:
            output = output.squeeze(-1)
        
        return output


class AdaptiveGatedFusion(nn.Module):
    def __init__(self, node_n, hidden_dim=512):
        super().__init__()
        self.node_n = node_n
        
        self.gate_net = nn.Sequential(
            nn.Linear(node_n * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim // 2, node_n),
            nn.Sigmoid()
        )
        
        self.context_net = nn.Sequential(
            nn.Linear(node_n * 2, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, node_n)
        )
        
        self.residual_transform = nn.Sequential(
            nn.Linear(node_n, node_n),
            nn.LayerNorm(node_n)
        )
        
    def forward(self, gcn_feat, tcn_feat):
        if len(gcn_feat.shape) == 4:
            B, T, V, C = gcn_feat.shape
            gcn_feat = gcn_feat.view(B, T, V*C)
            tcn_feat = tcn_feat.view(B, T, V*C)
        
        concat = torch.cat([gcn_feat, tcn_feat], dim=-1)
        
        gate = self.gate_net(concat)
        
        context = self.context_net(concat)
        
        residual = self.residual_transform(gcn_feat + tcn_feat)
        
        fused = gate * gcn_feat + (1 - gate) * tcn_feat + context + 0.1 * residual
        
        return fused


class DAST(nn.Module):
    def __init__(self, 
                 input_feature=3,
                 hidden_dim=256,
                 num_channels=[64, 128, 256],
                 kernel_size=3,
                 input_n=25,
                 output_n=100,
                 node_n=13,
                 dropout=0.1,
                 num_attention_heads=8):
        
        super().__init__()
        self.input_n = input_n
        self.output_n = output_n
        self.node_n = node_n
        self.hidden_dim = hidden_dim
        
        self.input_encoder = nn.Sequential(
            nn.Linear(input_feature, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.temporal_pos_encoding = nn.Parameter(torch.zeros(1, input_n + output_n, 1, hidden_dim))
        self.spatial_pos_encoding = nn.Parameter(torch.zeros(1, 1, node_n, hidden_dim))
        
        self.hierarchical_gcn = HierarchicalGCN(
            input_feature=hidden_dim,
            hidden_feature=hidden_dim,
            node_n=node_n,
            num_layers=3,
            dropout=dropout
        )
        
        self.multi_scale_tcn = MultiScaleTCN(
            num_inputs=hidden_dim,
            num_channels=num_channels,
            kernel_sizes=[2, 3, 5],
            dropout=dropout
        )
        
        self.st_attention = SpatioTemporalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        self.channel_attention = EnhancedSELayer(
            channel=hidden_dim,
            reduction=8,
            use_max_pooling=True
        )
        
        self.fusion_module = AdaptiveGatedFusion(
            node_n=node_n * hidden_dim,
            hidden_dim=hidden_dim * 2
        )
        
        self.time_expansion = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, input_feature)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
        
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def gen_velocity(self, motion):
        diffs = motion[:, 1:] - motion[:, :-1]
        last_velocity = diffs[:, -1:, :]
        velocity = torch.cat([diffs, last_velocity], dim=1)
        return velocity
    
    def forward(self, x):
        B, T_in, V3 = x.shape
        V = V3 // 3
        x_reshaped = x.view(B, T_in,V, 3)
        x_encoded = self.input_encoder(x_reshaped)
        
        x_encoded = x_encoded + self.temporal_pos_encoding[:, :T_in] + self.spatial_pos_encoding

        st_features = self.st_attention(x_encoded)
        st_features = self.dropout(st_features)

        gcn_input = st_features.view(B, T_in, V * self.hidden_dim)
        gcn_features = self.hierarchical_gcn(gcn_input)
        gcn_features = gcn_features.view(B, T_in, V, -1)
        
        velocity = self.gen_velocity(x.view(B, T_in, V, 3))
        velocity_encoded = self.input_encoder(velocity)
        
        tcn_input = velocity_encoded.permute(0, 3, 1, 2).contiguous()
        tcn_input = tcn_input.view(B, self.hidden_dim, T_in * V)
        
        tcn_features = self.multi_scale_tcn(tcn_input)
        tcn_features = tcn_features.view(B, self.hidden_dim, T_in, V)
        tcn_features = tcn_features.permute(0, 2, 3, 1).contiguous()
        
        gcn_features_perm = gcn_features.permute(0, 3, 1, 2)
        gcn_features_att = self.channel_attention(gcn_features_perm)
        gcn_features_att = gcn_features_att.permute(0, 2, 3, 1)

        tcn_features_perm = tcn_features.permute(0, 3, 1, 2)
        tcn_features_att = self.channel_attention(tcn_features_perm)
        tcn_features_att = tcn_features_att.permute(0, 2, 3, 1)

        gcn_flat = gcn_features_att.view(B, T_in, -1)
        tcn_flat = tcn_features_att.view(B, T_in, -1)
        fused_features = self.fusion_module(gcn_flat, tcn_flat)


        fused_features = fused_features.view(B, T_in, V, -1)
        
    
        fused_time_proj = fused_features.permute(0, 2, 1, 3).contiguous()
        B, V, T_in, C = fused_time_proj.shape
        fused_time_proj = fused_time_proj.view(B*V, T_in, C)
        
        time_exp_input = fused_features.permute(0, 2, 3, 1).contiguous()
        B_exp, V_exp, C_exp, T_exp = time_exp_input.shape
        
        time_exp_input = time_exp_input.view(B_exp * V_exp, C_exp, T_exp)
        
        time_extended = self.time_expansion(time_exp_input)
        
        _, C_out, T_out = time_extended.shape
        time_extended = time_extended.view(B_exp, V_exp, C_out, T_out)
        time_extended = time_extended.permute(0, 3, 1, 2).contiguous()
        
        predictions = self.prediction_head(time_extended)
        predictions = predictions.view(B, self.output_n, V * 3)
        
        return predictions
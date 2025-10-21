# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
# @Description : Network architecture for StackMFF-V3. This implementation includes:
# 1. PFMLP backbone with pyramid feature extraction
# 2. Depth-wise transformer for layer interaction modeling
# 3. Focus map creation for multi-focus image fusion
# 4. Flexible architecture supporting variable input stack sizes

import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
import math

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class ConvX(nn.Module):
    """Standard convolution block with normalization and activation."""
    def __init__(self, in_planes, out_planes, groups=1, kernel_size=3, stride=1, use_act=True):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=groups, padding=kernel_size//2, bias=False)
        self.norm = nn.BatchNorm2d(out_planes)
        self.act = nn.GELU() if use_act else nn.Identity()

    def forward(self, x):
        out = self.norm(self.conv(x))
        out = self.act(out)
        return out

class LearnablePool2d(nn.Module):
    """Learnable pooling layer with learnable weights."""
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dim = dim
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(1, 1, kernel_size, kernel_size), requires_grad=True)
        nn.init.normal_(self.weight, 0, 0.01)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        weight = self.weight.repeat(self.dim, 1, 1, 1)
        out = nn.functional.conv2d(x, weight, None, self.stride, self.padding, groups=self.dim)
        return self.norm(out)

class ChannelLearnablePool2d(nn.Module):
    """Channel-wise learnable pooling using depthwise convolution."""
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, groups=dim, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = self.conv(x)
        return self.norm(out)

class PyramidFC(nn.Module):
    """Pyramid feature combination module with multi-scale pooling."""
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, use_dw=False):
        super(PyramidFC, self).__init__()
        if use_dw:
            block = ChannelLearnablePool2d
        else:
            block = LearnablePool2d

        self.branch_1 = nn.Sequential(
            block(inplanes, kernel_size=3, stride=1, padding=1),
            ConvX(inplanes, planes, groups=1, kernel_size=1, use_act=False)
        )
        self.branch_2 = nn.Sequential(
            block(inplanes, kernel_size=5, stride=2, padding=2),
            ConvX(inplanes, planes, groups=1, kernel_size=1, use_act=False)
        )
        self.branch_3 = nn.Sequential(
            block(inplanes, kernel_size=7, stride=3, padding=3),
            ConvX(inplanes, planes, groups=1, kernel_size=1, use_act=False)
        )
        self.branch_4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvX(inplanes, planes, groups=1, kernel_size=1, use_act=False)
        )
        self.act = nn.GELU()

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = self.branch_1(x)
        x2 = nn.functional.interpolate(self.branch_2(x), size=(h, w), scale_factor=None, mode='nearest')
        x3 = nn.functional.interpolate(self.branch_3(x), size=(h, w), scale_factor=None, mode='nearest')
        x4 = self.branch_4(x)
        out = self.act(x1 + x2 + x3 + x4)
        return out

class BottleNeck(nn.Module):
    """Bottleneck residual block with spatial and channel MLP branches."""
    def __init__(self, in_planes, out_planes, stride=1, expand_ratio=1.0, mlp_ratio=1.0, use_dw=False, drop_path=0.0):
        super(BottleNeck, self).__init__()
        if use_dw:
            block = ChannelLearnablePool2d
        else:
            block = LearnablePool2d
        expand_planes = int(in_planes*expand_ratio)
        mid_planes = int(out_planes*mlp_ratio)

        self.smlp = nn.Sequential(
            PyramidFC(in_planes, expand_planes, kernel_size=3, stride=stride, use_dw=use_dw),
            ConvX(expand_planes, in_planes, groups=1, kernel_size=1, stride=1, use_act=False)
        )
        self.cmlp = nn.Sequential(
            ConvX(in_planes, mid_planes, groups=1, kernel_size=1, stride=1, use_act=True),
            block(mid_planes, kernel_size=3, stride=stride, padding=1) if stride==1 else ConvX(mid_planes, mid_planes, groups=mid_planes, kernel_size=3, stride=2, use_act=False),
            ConvX(mid_planes, out_planes, groups=1, kernel_size=1, stride=1, use_act=False)
        )

        self.skip = nn.Identity()
        # Downsampling is needed when stride==2
        if stride == 2:
            if in_planes != out_planes:
                self.skip = nn.Sequential(
                    ConvX(in_planes, in_planes, groups=in_planes, kernel_size=3, stride=2, use_act=False),
                    ConvX(in_planes, out_planes, groups=1, kernel_size=1, stride=1, use_act=False)
                )
            else:
                # When input and output channels are the same but downsampling is needed
                self.skip = ConvX(in_planes, out_planes, groups=in_planes, kernel_size=3, stride=2, use_act=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.drop_path(self.smlp(x)) + x
        x = self.drop_path(self.cmlp(x)) + self.skip(x)
        return x

class PyramidPoolingModule(nn.Module):
    """Pyramid Pooling Module from UPerNet"""
    
    def __init__(self, in_channels, out_channels):
        super(PyramidPoolingModule, self).__init__()
        inter_channels = in_channels // 4
        self.conv1 = ConvX(in_channels, inter_channels, kernel_size=1, stride=1, use_act=True)
        self.conv2 = ConvX(in_channels, inter_channels, kernel_size=1, stride=1, use_act=True)
        self.conv3 = ConvX(in_channels, inter_channels, kernel_size=1, stride=1, use_act=True)
        self.conv4 = ConvX(in_channels, inter_channels, kernel_size=1, stride=1, use_act=True)
        self.out = ConvX(in_channels * 2, out_channels, kernel_size=1, stride=1, use_act=True)

    def pool(self, x, size):
        return nn.AdaptiveAvgPool2d(size)(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode="bilinear", align_corners=False)

    def forward(self, x):
        size = x.shape[2:]
        f1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        f2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        f3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        f4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        f = torch.cat([x, f1, f2, f3, f4], dim=1)
        return self.out(f)

class PFMLP(nn.Module):
    """Pyramid Feature MLP Network backbone."""
    def __init__(self, dims, layers, block=BottleNeck, expand_ratio=1.0, mlp_ratio=1.0, use_dw=False, drop_path_rate=0., num_classes=1000):
        super(PFMLP, self).__init__()
        self.block = block
        self.expand_ratio = expand_ratio
        self.mlp_ratio = mlp_ratio
        self.use_dw = use_dw
        self.drop_path_rate = drop_path_rate

        if isinstance(dims, int):
            dims = [dims//2, dims, dims*2, dims*4, dims*8]
        else:
            dims = [dims[0]//2] + dims #[8,16,24,32,48]

        self.first_conv = ConvX(1, dims[0], 1, 3, 2, use_act=True)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]

        self.layer1 = self._make_layers(dims[0], dims[1], layers[0], stride=2, drop_path=dpr[:layers[0]])
        self.layer2 = self._make_layers(dims[1], dims[2], layers[1], stride=2, drop_path=dpr[layers[0]:sum(layers[:2])])
        self.layer3 = self._make_layers(dims[2], dims[3], layers[2], stride=2, drop_path=dpr[sum(layers[:2]):sum(layers[:3])])
        self.layer4 = self._make_layers(dims[3], dims[4], layers[3], stride=2, drop_path=dpr[sum(layers[:3]):sum(layers[:4])])

        # UPerNet-style decoder replacing the previous upsample decoder
        # PPM on top feature map
        self.ppm = PyramidPoolingModule(dims[4], dims[0])
        
        # Lateral connections to unify channels
        self.fpn_in = nn.ModuleDict({
            'fpn_layer1': ConvX(dims[1], dims[0], kernel_size=1, stride=1, use_act=True),
            'fpn_layer2': ConvX(dims[2], dims[0], kernel_size=1, stride=1, use_act=True),
            'fpn_layer3': ConvX(dims[3], dims[0], kernel_size=1, stride=1, use_act=True),
        })
        
        # FPN output smoothing
        self.fpn_out = nn.ModuleDict({
            'fpn_layer1': ConvX(dims[0], dims[0], kernel_size=3, stride=1, use_act=True),
            'fpn_layer2': ConvX(dims[0], dims[0], kernel_size=3, stride=1, use_act=True),
            'fpn_layer3': ConvX(dims[0], dims[0], kernel_size=3, stride=1, use_act=True),
        })
        
        # Fuse multi-scale features
        self.fuse = ConvX(dims[0]*4, dims[0], kernel_size=1, stride=1, use_act=True)
        self.final_conv = ConvX(dims[0], dims[0], kernel_size=3, stride=1, use_act=True)

        self.init_params(self)

    def _make_layers(self, inputs, outputs, num_block, stride, drop_path):
        layers = [self.block(inputs, outputs, stride, self.expand_ratio, self.mlp_ratio, self.use_dw, drop_path[0])]

        for i in range(1, num_block):
            layers.append(self.block(outputs, outputs, 1, self.expand_ratio, self.mlp_ratio, self.use_dw, drop_path[i]))
            
        return nn.Sequential(*layers)

    def init_params(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Store original input size for final upsampling
        input_size = x.shape[2:]
        
        # Encoder
        x0 = self.first_conv(x)  # 1/2
        x1 = self.layer1(x0)     # 1/4
        x2 = self.layer2(x1)     # 1/8
        x3 = self.layer3(x2)     # 1/16
        x4 = self.layer4(x3)     # 1/32
        
        # Decoder: UPerNet-style
        # PPM on top feature map
        ppm_out = self.ppm(x4)  # dims[0] channels

        # Top-down pathway
        f = ppm_out
        lat3 = self.fpn_in['fpn_layer3'](x3)
        f = F.interpolate(f, size=lat3.shape[2:], mode='bilinear', align_corners=False)
        f = lat3 + f
        fpn3 = self.fpn_out['fpn_layer3'](f)

        lat2 = self.fpn_in['fpn_layer2'](x2)
        f = F.interpolate(f, size=lat2.shape[2:], mode='bilinear', align_corners=False)
        f = lat2 + f
        fpn2 = self.fpn_out['fpn_layer2'](f)

        lat1 = self.fpn_in['fpn_layer1'](x1)
        f = F.interpolate(f, size=lat1.shape[2:], mode='bilinear', align_corners=False)
        f = lat1 + f
        fpn1 = self.fpn_out['fpn_layer1'](f)

        # Fuse features at 1/4 scale
        target_size = fpn1.shape[2:]
        feats = [
            fpn1,
            F.interpolate(fpn2, size=target_size, mode='bilinear', align_corners=False),
            F.interpolate(fpn3, size=target_size, mode='bilinear', align_corners=False),
            F.interpolate(ppm_out, size=target_size, mode='bilinear', align_corners=False),
        ]
        fused = self.fuse(torch.cat(feats, dim=1))

        # Upsample to original resolution using input size (like UPerNet)
        out = F.interpolate(fused, size=input_size, mode='bilinear', align_corners=False)
        out = self.final_conv(out)
        return out

class FeatureEncoder(nn.Module):
    """Feature encoder module that processes input image stacks."""
    def __init__(self, dims=[16,32,64,128],layers=[2,2,6,2]):
        super(FeatureEncoder, self).__init__()
        self.dims=dims
        self.embbed_dim=dims[0]//2
        self.layers=layers
        self.encoder = PFMLP(dims=dims, layers=layers, expand_ratio=3.0, mlp_ratio=3.0, use_dw=False, drop_path_rate=0.10)
    
    def forward(self, x):
        batch_size, num_images, height, width = x.shape
        x_reshaped = x.view(batch_size * num_images, 1, height, width)
        out = self.encoder(x_reshaped)
        # focus_maps_single_layer = torch.nn.functional.silu(out)
        focus_maps_single_layer = torch.nn.functional.sigmoid(out)

        return focus_maps_single_layer.view(batch_size, num_images, self.embbed_dim, height, width)

def apply_rotary_pos_emb(x, sin, cos):
    """
    Apply Rotary Positional Encoding (RoPE) to features
    x: [..., N, C]  # N is num_images, C is embed_dim
    sin, cos: [N, C//2]
    Output: [..., N, C]
    """
    x1 = x[..., ::2]  # Even dimensions
    x2 = x[..., 1::2] # Odd dimensions
    # Apply rotation transformation to each pair (x1, x2)
    x = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    x = x.flatten(-2)  # Restore original embed_dim dimension
    return x

def get_rotary_emb(dim, seq_len, device):
    """
    Generate Rotary Positional Encoding sin, cos
    dim: embed_dim
    seq_len: depth direction length (num_images)
    Returns:
        sin, cos: [seq_len, dim//2]
    """
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(seq_len, device=device).float()
    sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)  # [seq_len, dim//2]
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    return sin, cos

class DepthTransformerLayer(nn.Module):
    """
    Single layer Depth Transformer Layer
    """
    def __init__(self, embed_dim, num_heads, ff_dim=None, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Multi-head attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff_dim = ff_dim or embed_dim * 4
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, self.ff_dim),
            nn.GELU(),
            nn.Linear(self.ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B*H*W, N, C]
        """
        # Rotary Positional Encoding
        sin, cos = get_rotary_emb(self.embed_dim, x.size(1), x.device)  # [N, C//2]
        x_rot = apply_rotary_pos_emb(x, sin, cos)  # [B*H*W, N, C]

        # Self-attention along depth direction
        x_attn = self.attn(x_rot, x_rot, x_rot)[0]  # [B*H*W, N, C]
        x_attn = self.dropout(x_attn)

        # Residual connection + LayerNorm
        x = self.norm1(x_rot + x_attn)  # [B*H*W, N, C]

        # Feed-forward network
        x_ffn = self.ffn(x)  # [B*H*W, N, C]
        x_ffn = self.dropout(x_ffn)
        x = self.norm2(x + x_ffn)  # [B*H*W, N, C]

        return x

class DepthTransformer(nn.Module):
    """
    Transformer along depth direction (num_images) to capture inter-layer relationships
    Input and output are both [B, N, C, H, W]
    Supports custom number of layers
    """
    def __init__(self, embed_dim, num_heads, num_layers=1, ff_dim=None, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Create multiple transformer layers
        self.layers = nn.ModuleList([
            DepthTransformerLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: [B, N, C, H, W]
        """
        B, N, C, H, W = x.shape

        # Flatten each spatial position to batch dimension
        # Original x: [B, N, C, H, W]
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B*H*W, N, C)
        # [B*H*W, N, C] -> Each spatial position independently performs depth direction attention

        # Pass through multiple transformer layers
        for layer in self.layers:
            x_flat = layer(x_flat)

        # Restore original spatial structure
        x = x_flat.reshape(B, H, W, N, C).permute(0, 3, 4, 1, 2)  # [B, N, C, H, W]

        return x

class LayerInteraction(nn.Module):
    """Layer Interaction Module
    Uses bidirectional ConvGRU for temporal feature interaction
    """
    def __init__(self, embed_dim, num_transformer_layers=1):
        super(LayerInteraction, self).__init__()
        self.layer_interaction_depth = DepthTransformer(embed_dim=embed_dim, num_heads=4, num_layers=num_transformer_layers)

        # self.proj_pool_depth = nn.AdaptiveMaxPool3d(output_size=(1, None, None))
        # Use fixed-size MaxPool3d instead of AdaptiveMaxPool3d
        self.proj_pool_depth = nn.MaxPool3d(kernel_size=(embed_dim, 1, 1), stride=(embed_dim, 1, 1))

    def forward(self, focus_maps):
        batch_size, num_images, embed_dim, height, width = focus_maps.shape

        att_out = self.layer_interaction_depth(focus_maps)#[B,N,C,H,W]
 
        # Apply MaxPool3d to reduce channel dimension: [B,N,C,H,W] -> [B,N,1,H,W]
        focus_maps_depth = self.proj_pool_depth(att_out)
        focus_maps_depth = focus_maps_depth.squeeze(2)  # Remove channel dimension: [B,N,H,W]

        return focus_maps_depth
        
class FocusMapCreation(nn.Module):
    """Focus Map Creation Module
    Creates focus index map from focus maps using softmax
    """
    def __init__(self):
        super(FocusMapCreation, self).__init__()

    def forward(self, focus_maps_depth, num_images):
        # Step 1: Calculate focus probabilities
        focus_probs = F.softmax(focus_maps_depth, dim=1)
        
        # Step 2: Find the layer index with maximum probability (0 to N-1)
        focus_map = torch.argmax(focus_probs, dim=1, keepdim=True).float()  # [B, 1, H, W]
        
        return focus_map
        
class StackMFF_V3(nn.Module):
    """StackMFF-V3: Multi-focus image fusion network with transformer-based layer interaction modeling."""
    def __init__(self, encoder_dims=[32,64,128,256], encoder_layers=[2,2,2,2], num_transformer_layers=2):
        super(StackMFF_V3, self).__init__()
        # Calculate embedding dimension, no need to store as instance variable
        encoder_embed_dim = encoder_dims[0] // 2
        
        self.feature_extraction = FeatureEncoder(dims=encoder_dims, layers=encoder_layers) 
        self.layer_interaction = LayerInteraction(embed_dim=encoder_embed_dim, num_transformer_layers=num_transformer_layers)
        self.focus_map_creation = FocusMapCreation()
        
    def forward(self, x):
        """Forward propagation - supports multi-class tasks with variable number of classes
        
        Args:
            x: Input image stack [batch_size, num_images, height, width]
            Note: num_images is the number of classes, which can vary between batches
            
        Returns:
            if self.training (training mode):
                layer_interaction_features: [batch_size, num_images, height, width]
                These are logits for multi-class classification, number of classes = num_images
            else (inference mode):
                fused_image: [batch_size, 1, height, width]
                focus_indices: [batch_size, height, width] value range [0, num_images-1]
        """
        batch_size, num_images, height, width = x.shape
        
        # Check input validity
        assert num_images >= 2, f"Number of images must be at least 2, current is {num_images}"
        
        respective_features = self.feature_extraction(x) # Shape: [batch_size, num_images, embed_dim, height, width]
        layer_interaction_features = self.layer_interaction(respective_features) # Shape: [batch_size, num_images, height, width]

        if self.training:
            # Training mode: Return logits for multi-class classification, number of classes = num_images
            # This output will be used by F.cross_entropy, supporting variable number of classes
            return layer_interaction_features
        else:
            # Inference mode: Return fused image and focus indices
            focus_map = self.focus_map_creation(layer_interaction_features, num_images) # Shape: [batch_size, 1, height, width]
            fused_image, focus_indices = self.generate_fused_image(x, focus_map)
            return fused_image, focus_indices

    def generate_fused_image(self, x, focus_map):
        """
        Generate fused image using focus index map - supports variable number of classes
        
        Args:
            x: Input image stack [batch_size, num_images, height, width]
            focus_map: Focus index map [batch_size, 1, height, width] value range [0, num_images-1]
            
        Returns:
            fused_image: Final fused image [batch_size, 1, height, width]
            focus_indices: Focus indices [batch_size, height, width] value range [0, num_images-1]
        """
        batch_size, num_images, height, width = x.shape
        focus_indices = focus_map.squeeze(1).long()  # [batch_size, height, width]
        
        # Ensure indices are within valid range
        focus_indices = torch.clamp(focus_indices, 0, num_images - 1)
        
        # Use torch.gather to select pixel values by index
        fused_image = torch.gather(x, dim=1, index=focus_indices.unsqueeze(1))
        
        return fused_image, focus_indices
        
    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, BidirectionalConvGRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

if __name__ == "__main__":
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model and move to device
    model = StackMFF_V3().to(device)
    
    # Create input data
    x = torch.randn(1, 5, 256, 256).to(device)
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        fused_image, focus_indices = model(x)
    
    # Print output shapes
    print(f"Input shape: {x.shape}")
    print(f"Fused image shape: {fused_image.shape}")
    print(f"Focus indices shape: {focus_indices.shape}")
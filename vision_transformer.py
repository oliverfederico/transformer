import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict


# class MLPBlock(torchvision.ops.MLP): # currently not used
#     def __init__(self, hidden_dim: int, mlp_dim: int):
#         super().__init__(
#             hidden_dim,
#             mlp_dim,
#             activation_layer=nn.GELU,
#             dropout=0.1,
#         )


class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout
        )
        self.dropout = nn.Dropout(p=dropout)

        self.ln_2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = torchvision.ops.MLP(
            hidden_dim,
            [mlp_dim, hidden_dim],
            activation_layer=nn.GELU,
            dropout=dropout,
        )

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 2,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.empty(seq_length, hidden_dim).normal_(std=0.02)
        )  # from BERT
        self.dropout = nn.Dropout(p=dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()

        for i in range(num_layers):
            layers[f"layer_{i}"] = EncoderBlock(hidden_dim, num_heads, mlp_dim, dropout)
        self.layers = nn.Sequential(layers)
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 2,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        x = input + self.pos_embedding
        x = self.dropout(x)
        x = self.layers(x)
        x = self.ln(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv_proj = nn.Conv2d(
            in_channels=1,  # for grayscale images
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        seq_length = (image_size // patch_size) ** 2

        # add a class token
        self.class_token = nn.Parameter(torch.zeros(1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
        )

        self.seq_length = seq_length

        self.heads = nn.Linear(hidden_dim, num_classes)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape
        p = self.patch_size
        torch._assert(
            h == self.image_size,
            f"Wrong image height! Expected {self.image_size} but got {h}!",
        )
        torch._assert(
            w == self.image_size,
            f"Wrong image width! Expected {self.image_size} but got {w}!",
        )
        n_h = h // p
        n_w = w // p

        # (h, w) -> (hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (hidden_dim, n_h, n_w) -> (hidden_dim, (n_h * n_w))
        x = x.reshape(self.hidden_dim, n_h * n_w)

        # (hidden_dim, (n_h * n_w)) -> ((n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (S, E)
        # where S is the source sequence length, E is the
        # embedding dimension
        x = x.permute(1, 0)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)

        # prepend the class token
        x = torch.cat([self.class_token.view(1, -1), x])

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[0]

        x = self.heads(x)

        return x

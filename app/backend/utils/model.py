import torch
import torch.nn as nn
import torch.nn.functional as F


class BehaviorRNN(nn.Module):
    def __init__(
        self,
        rnn_type: str = "BiLSTM",
        input_size: int = 563,  # 512 CNN + 11 Motion/BBox + 40 Keypoints
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        geom_dim: int = 11,     # Motion + Keypoints
        visual_hidden: int = 128,
        geom_hidden: int = 64,
    ):
        super().__init__()
        self.geom_dim = geom_dim
        visual_dim = input_size - geom_dim

        # Feature projection MLPs
        self.visual_mlp = nn.Sequential(
            nn.Linear(visual_dim, visual_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(visual_hidden, visual_hidden),
            nn.ReLU(inplace=True),
        )
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_dim, geom_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(geom_hidden, geom_hidden),
            nn.ReLU(inplace=True),
        )

        # BiLSTM Sequence Modeling
        self.rnn = nn.LSTM(
            input_size=visual_hidden + geom_hidden,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        # Attention Pooling & Classifier
        rnn_out_dim = hidden_size * 2
        self.attn = nn.Linear(rnn_out_dim, 1)
        self.fc = nn.Linear(rnn_out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split visual and geometry/keypoint features
        v_feat = self.visual_mlp(x[..., :-self.geom_dim])
        g_feat = self.geom_mlp(x[..., -self.geom_dim:])
        
        # Temporal processing
        out, _ = self.rnn(torch.cat([v_feat, g_feat], dim=-1))

        # Attention aggregation over sequence
        attn_weights = F.softmax(self.attn(out), dim=1)
        context = (attn_weights * out).sum(dim=1)
        
        return self.fc(context)
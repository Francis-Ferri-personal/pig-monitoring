import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualBehaviorRNN(nn.Module):
    def __init__(
        self,
        rnn_type: str = "LSTM",
        input_size: int = 512 + 11,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        geom_dim: int = 11,
        visual_hidden: int = 128,
        geom_hidden: int = 64,
    ):
        """
        Recurrent model for visual embeddings + bbox/motion features.

        Args:
            rnn_type (str): 'RNN', 'GRU', 'LSTM' or 'BiLSTM'
            input_size (int): Total features per frame (CNN emb + geometry + motion).
            hidden_size (int): Size of the RNN hidden state (per direction if bidirectional).
            num_layers (int): Number of recurrent layers.
            num_classes (int): Number of output actions.
            geom_dim (int): Number of geometric/motion features at the end of the vector.
            visual_hidden (int): Hidden size of visual MLP branch.
            geom_hidden (int): Hidden size of geometric/motion MLP branch.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.upper()
        self.bidirectional = self.rnn_type == "BILSTM"

        # Split input into visual and geometric parts
        self.geom_dim = geom_dim
        self.visual_dim = input_size - geom_dim

        # Two-branch MLPs to balance representation sizes
        self.visual_mlp = nn.Sequential(
            nn.Linear(self.visual_dim, visual_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(visual_hidden, visual_hidden),
            nn.ReLU(inplace=True),
        )
        self.geom_mlp = nn.Sequential(
            nn.Linear(self.geom_dim, geom_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(geom_hidden, geom_hidden),
            nn.ReLU(inplace=True),
        )

        rnn_input_dim = visual_hidden + geom_hidden

        if self.rnn_type in {"LSTM", "BILSTM"}:
            self.rnn = nn.LSTM(
                rnn_input_dim,
                hidden_size,
                num_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=0.2 if num_layers > 1 else 0.0,
            )
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                rnn_input_dim,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0,
            )
        elif self.rnn_type == "RNN":
            self.rnn = nn.RNN(
                rnn_input_dim,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unknown rnn_type: {self.rnn_type}. Use 'RNN', 'GRU', 'LSTM', or 'BiLSTM'.")

        # Output dimension of RNN (account for bidirectionality)
        rnn_out_dim = hidden_size * (2 if self.bidirectional else 1)

        # Attention pooling: enabled by default for BiLSTM
        self.use_attention = self.bidirectional
        if self.use_attention:
            self.attn = nn.Linear(rnn_out_dim, 1)

        self.fc = nn.Linear(rnn_out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        # Split into visual and geometric parts
        visual = x[..., :-self.geom_dim]
        geom = x[..., -self.geom_dim :]

        v_feat = self.visual_mlp(visual)
        g_feat = self.geom_mlp(geom)
        fused = torch.cat([v_feat, g_feat], dim=-1)

        out, _ = self.rnn(fused)

        if self.use_attention:
            attn_scores = self.attn(out)
            attn_weights = F.softmax(attn_scores, dim=1)
            context = (attn_weights * out).sum(dim=1)
            logits = self.fc(context)
        else:
            last_frame_out = out[:, -1, :]
            logits = self.fc(last_frame_out)
        return logits


if __name__ == "__main__":
    for r_type in ["RNN", "GRU", "LSTM", "BiLSTM"]:
        model = VisualBehaviorRNN(rnn_type=r_type, input_size=523, hidden_size=128, num_classes=5)
        dummy_input = torch.randn(8, 60, 523)
        output = model(dummy_input)
        print(f"[{r_type}] Output shape:", output.shape)
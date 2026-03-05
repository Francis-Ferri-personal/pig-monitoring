import torch
import torch.nn as nn

class BehaviorRNN(nn.Module):
    def __init__(self, rnn_type='LSTM', input_size=57, hidden_size=64, num_layers=2, num_classes=5):
        """
        Architecture selection for pig behavior recognition.
        Args:
            rnn_type (str): 'RNN', 'GRU', or 'LSTM'
            input_size (int): Number of features per frame.
            hidden_size (int): Size of the hidden state.
            num_layers (int): Number of recurrent layers.
            num_classes (int): Number of output actions.
        """
        super(BehaviorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.upper()

        # Recurrent Layer Selection
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        else:
            raise ValueError(f"Unknown rnn_type: {self.rnn_type}. Use 'RNN', 'GRU', or 'LSTM'.")

        # Fully Connected Classifier
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        
        # Initialize hidden state
        # In PyTorch, LSTM returns (out, (h, c)), GRU/RNN return (out, h)
        if self.rnn_type == 'LSTM':
            out, _ = self.rnn(x) # (h0, c0) are initialized to zero by default
        else:
            out, _ = self.rnn(x) # (h0) initialized to zero by default
            
        # Decode the hidden state of the last time step
        # out shape: (batch, seq_len, hidden_size)
        last_frame_out = out[:, -1, :] 
        
        # Final classification
        logits = self.fc(last_frame_out)
        return logits

if __name__ == "__main__":
    # Test all three types
    for r_type in ['RNN', 'GRU', 'LSTM']:
        model = BehaviorRNN(rnn_type=r_type, input_size=57, hidden_size=64, num_classes=4)
        dummy_input = torch.randn(8, 30, 57) # batch=8, seq=30, features=57
        output = model(dummy_input)
        print(f"[{r_type}] Output shape:", output.shape) # Expected: [8, 4]

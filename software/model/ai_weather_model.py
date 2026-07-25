import torch
import torch.nn as nn

"""
    ConvLSTM replaces matrix multiplications with 2D convolutions, enabling the cell 
    to retain 2D spatial structures (like weather maps) while tracking temporal state.
"""
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()

        # Calculates padding dynamically to keep the spatial dimensions (H, W) identical across convolutions
        padding = kernel_size // 2

        """
        # Concatenates input (x) and previous hidden state (h_prev) along the channel axis.
        To compute all four LSTM gates simultaneously, we project this combined representation
        into 4 * hidden_channels in a single convolution step to maximize GPU execution speed.
        """
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.hidden_channels = hidden_channels

    # Goes a step forward in time
    def forward(self, x, state):

        h_prev, c_prev = state

        # Shape: (B, C_in + C_hidden, H, W)
        combined = torch.cat([x, h_prev], dim=1)

        # Shape: (B, 4 * C_hidden, H, W)
        gates = self.conv(combined)

        # Slice the tensor into 4 equal chunks along the channel dimension
        # i: Input, f: Forget, o: Output, g: Cell input (candidate)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        # Apply standard LSTM non-linear gating activations
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        # Update cell state (c) and hidden state (h)
        # c = (Forget * History) + (Input * Candidate Updates)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c
    
    # Initialises empty cell with zeroes
    def init_state(self, batch_size, height, width, device, dtype):
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device, dtype=dtype)
        return h, c


class ConvLSTMStack(nn.Module):
    """Stacks multiple ConvLSTMCell layers, feeding each layer's hidden
    state as input to the next layer."""

    def __init__(self, in_channels, hidden_channels_list, kernel_size=3):
        super().__init__()
        self.hidden_channels_list = hidden_channels_list
        cells = []
        prev_channels = in_channels
        for hidden_channels in hidden_channels_list:
            cells.append(ConvLSTMCell(prev_channels, hidden_channels, kernel_size))
            prev_channels = hidden_channels
        # nn.ModuleList (not a plain list) so PyTorch registers the
        # parameters of every cell for .to(device), .parameters(), etc.
        self.cells = nn.ModuleList(cells)

    def init_states(self, batch_size, height, width, device, dtype):
        return [
            cell.init_state(batch_size, height, width, device, dtype)
            for cell in self.cells
        ]

    def forward_step(self, x, states):
        new_states = []
        layer_input = x
        for cell, state in zip(self.cells, states):
            h, c = cell(layer_input, state)
            new_states.append((h, c))
            layer_input = h
        return layer_input, new_states


class NowcastNet(nn.Module):
    """
    The main hybrid Physics-Residual Nowcasting neural network.
    
    Instead of predicting the complex physics of rain directly, this model uses a 
    stacked ConvLSTM to predict a "residual change" map. It then adds this residual map
    to a classical advection prior projection to generate physically anchored predictions.
    """

    def __init__(self, dynamic_channels=2, static_channels=3,
                 hidden_channels=(32, 64, 64), forecast_steps=6, kernel_size=3, residual_scale=1.0):
        super().__init__()
        self.residual_scale = residual_scale
        in_channels = dynamic_channels + static_channels
        self.dynamic_channels = dynamic_channels
        self.static_channels = static_channels
        self.forecast_steps = forecast_steps

        # Multi-layer deep spatio-temporal features processor
        self.stack = ConvLSTMStack(in_channels, list(hidden_channels), kernel_size)

        # Decoder Head: Resolves hidden features + 1D advection prior back into dynamic outputs
        last_hidden = hidden_channels[-1]
        self.decoder_head = nn.Sequential(
            nn.Conv2d(last_hidden + 1, last_hidden // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_hidden // 2, dynamic_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, static, forecast_steps=None, advection_prior=None, return_all_channels=False):
        """
        Args:
            x: Historical dynamic frames. Shape: (B, T_in, C_dynamic, H, W)
            static: Static geographic masks. Shape: (B, C_static, H, W)
            forecast_steps: Number of future predictions. Defaults to self.forecast_steps.
            advection_prior: Physics projections from advection.py. Shape: (B, T_out, 1, H, W)
        """

        forecast_steps = forecast_steps or self.forecast_steps
        batch_size, t_in, _, height, width = x.shape
        device, dtype = x.device, x.dtype

        if advection_prior is None:
            advection_prior = torch.zeros(
                batch_size, forecast_steps, 1, height, width, device=device, dtype=dtype
            )
        
        # Initialises hidden states for all layers in ConvLSTM stack
        states = self.stack.init_states(batch_size, height, width, device, dtype)

        # Passes historical dynamic frames sequentially to build temporal context inside recurrent memory cells.
        for t in range(t_in):
            frame = torch.cat([x[:, t], static], dim=1)
            _, states = self.stack.forward_step(frame, states)

        outputs = []
        last_frame = x[:, -1]

        # Assembles future frames with physics integration
        for step in range(forecast_steps):
            frame_in = torch.cat([last_frame, static], dim=1)
            top_h, states = self.stack.forward_step(frame_in, states)

            prior_t = advection_prior[:, step]  # (B, 1, H, W)
            decoder_in = torch.cat([top_h, prior_t], dim=1)
            residual = self.decoder_head(decoder_in)  # (B, dynamic_channels, H, W)

            pred = residual.clone()
            pred[:, 0:1] = torch.clamp(self.residual_scale * residual[:, 0:1] + prior_t, 0.0, 1.0)  # radar = prior + residual
            pred[:, 1:] = torch.sigmoid(residual[:, 1:])  # non-radar channels (satellite) have no prior

            outputs.append(pred)
            last_frame = pred  # feed prediction back in as next input

        outputs = torch.stack(outputs, dim=1)  # (B, T_out, dynamic_channels, H, W)
        if return_all_channels:
            return outputs  # (B, T_out, dynamic_channels, H, W) -- radar + satellite/IR
        # Radar is channel 0 of the dynamic channels -- that's what we nowcast.
        radar_pred = outputs[:, :, 0:1]  # (B, T_out, 1, H, W)
        return radar_pred


def weighted_rain_mse(pred, target, weight_power=2.0, base_weight=1.0):
    weight = base_weight + target.clamp(min=0.0) ** weight_power
    return (weight * (pred - target) ** 2).mean()


if __name__ == "__main__":
    # quick shape sanity check (runs on CPU with random data, no real data needed)
    model = NowcastNet(dynamic_channels=2, static_channels=2,
                        hidden_channels=(16, 32), forecast_steps=6)
    x = torch.rand(2, 4, 2, 64, 64)      # batch=2, T_in=4, C=2, 64x64 patch
    static = torch.rand(2, 2, 64, 64)    # batch=2, static C=2, 64x64 patch (matches example below)
    out = model(x, static)  # no prior supplied -> defaults to zeros
    print("Output shape (no prior):", out.shape)

    prior = torch.rand(2, 6, 1, 64, 64)  # dummy advection prior, e.g. from advection.py
    out_with_prior = model(x, static, advection_prior=prior)
    print("Output shape (with prior):", out_with_prior.shape)
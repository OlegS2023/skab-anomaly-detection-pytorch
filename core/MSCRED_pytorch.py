# ============================================================
# MSCRED_pytorch.py
# Multi-Scale Convolutional Recurrent Encoder-Decoder
# Stabilna wersja PyTorch
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# ConvLSTM CELL
# ============================================================

class _ConvLSTM2DCell(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2
        self.out_channels = out_channels

        self.conv = nn.Conv2d(
            in_channels + out_channels,
            4 * out_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):

        combined = torch.cat([x, h], dim=1)

        gates = self.conv(combined)

        i, f, o, g = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# ============================================================
# ConvLSTM
# ============================================================

class _ConvLSTM2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.cell = _ConvLSTM2DCell(
            in_channels,
            out_channels,
            kernel_size
        )

        self.out_channels = out_channels

    def forward(self, x):

        # x -> (B, T, C, H, W)

        B, T, C, H, W = x.shape

        h = torch.zeros(
            B,
            self.out_channels,
            H,
            W,
            device=x.device
        )

        c = torch.zeros(
            B,
            self.out_channels,
            H,
            W,
            device=x.device
        )

        outputs = []

        for t in range(T):

            h, c = self.cell(x[:, t], h, c)

            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1)


# ============================================================
# Attention
# ============================================================

class _AttentionLayer(nn.Module):

    def __init__(self, step_max):
        super().__init__()

        self.step_max = step_max

    def forward(self, outputs):

        # outputs -> (B, T, C, H, W)

        B, T, C, H, W = outputs.shape

        last = outputs[:, -1]

        weights = []

        for t in range(T):

            score = (outputs[:, t] * last).sum(dim=(1, 2, 3))

            weights.append(score)

        weights = torch.stack(weights, dim=1)

        weights = torch.softmax(weights, dim=1)

        weights = weights.view(B, T, 1, 1, 1)

        attended = (outputs * weights).sum(dim=1)

        return attended


# ============================================================
# MSCRED NETWORK
# ============================================================

class _MSCREDNetwork(nn.Module):

    def __init__(self, sensor_n, scale_n, step_max):

        super().__init__()

        self.sensor_n = sensor_n
        self.scale_n = scale_n
        self.step_max = step_max

        # ====================================================
        # Padding do wielokrotności 8
        # ====================================================

        if sensor_n % 8 != 0:
            self.sensor_n_pad = (sensor_n // 8) * 8 + 8
        else:
            self.sensor_n_pad = sensor_n

        self.pad_size = self.sensor_n_pad - sensor_n

        # ====================================================
        # ENCODER
        # ====================================================

        self.conv1 = nn.Conv2d(
            scale_n,
            32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.conv4 = nn.Conv2d(
            128,
            256,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.act = nn.SELU()

        # ====================================================
        # ConvLSTM
        # ====================================================

        self.convlstm1 = _ConvLSTM2D(32, 32)
        self.convlstm2 = _ConvLSTM2D(64, 64)
        self.convlstm3 = _ConvLSTM2D(128, 128)
        self.convlstm4 = _ConvLSTM2D(256, 256)

        # ====================================================
        # Attention
        # ====================================================

        self.att1 = _AttentionLayer(step_max)
        self.att2 = _AttentionLayer(step_max)
        self.att3 = _AttentionLayer(step_max)
        self.att4 = _AttentionLayer(step_max)

        # ====================================================
        # DECODER
        # ====================================================

        self.deconv4 = nn.ConvTranspose2d(
            256 + 128,
            128,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.deconv3 = nn.ConvTranspose2d(
            128 + 64,
            64,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.deconv2 = nn.ConvTranspose2d(
            64 + 32,
            32,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.deconv1 = nn.Conv2d(
            32,
            scale_n,
            kernel_size=3,
            padding=1
        )

    # ========================================================
    # ENCODER
    # ========================================================

    def _encode_seq(self, x):

        B, T, C, H, W = x.shape

        x_flat = x.reshape(B * T, C, H, W)

        c1_flat = self.act(self.conv1(x_flat))
        c2_flat = self.act(self.conv2(c1_flat))
        c3_flat = self.act(self.conv3(c2_flat))
        c4_flat = self.act(self.conv4(c3_flat))

        c1 = c1_flat.view(B, T, 32,
                          c1_flat.shape[2],
                          c1_flat.shape[3])

        c2 = c2_flat.view(B, T, 64,
                          c2_flat.shape[2],
                          c2_flat.shape[3])

        c3 = c3_flat.view(B, T, 128,
                          c3_flat.shape[2],
                          c3_flat.shape[3])

        c4 = c4_flat.view(B, T, 256,
                          c4_flat.shape[2],
                          c4_flat.shape[3])

        return c1, c2, c3, c4

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # (B,T,H,W,C) -> (B,T,C,H,W)

        x = x.permute(0, 1, 4, 2, 3).float()

        B, T, C, H, W = x.shape

        # ====================================================
        # Padding
        # ====================================================

        if self.pad_size > 0:

            x = F.pad(
                x,
                (0, self.pad_size, 0, self.pad_size)
            )

        # ====================================================
        # Encoder
        # ====================================================

        c1, c2, c3, c4 = self._encode_seq(x)

        # ====================================================
        # ConvLSTM
        # ====================================================

        lstm1 = self.convlstm1(c1)
        lstm2 = self.convlstm2(c2)
        lstm3 = self.convlstm3(c3)
        lstm4 = self.convlstm4(c4)

        # ====================================================
        # Attention
        # ====================================================

        a1 = self.att1(lstm1)
        a2 = self.att2(lstm2)
        a3 = self.att3(lstm3)
        a4 = self.att4(lstm4)

        # ====================================================
        # DEBUG SHAPES
        # ====================================================

        # print(a1.shape)
        # print(a2.shape)
        # print(a3.shape)
        # print(a4.shape)

        # ====================================================
        # Decoder
        # ====================================================

        a4 = F.interpolate(
            a4,
            size=a3.shape[2:],
            mode='nearest'
        )

        d4 = self.act(
            self.deconv4(
                torch.cat([a4, a3], dim=1)
            )
        )

        d4 = F.interpolate(
            d4,
            size=a2.shape[2:],
            mode='nearest'
        )

        d3 = self.act(
            self.deconv3(
                torch.cat([d4, a2], dim=1)
            )
        )

        d3 = F.interpolate(
            d3,
            size=a1.shape[2:],
            mode='nearest'
        )

        d2 = self.act(
            self.deconv2(
                torch.cat([d3, a1], dim=1)
            )
        )

        d1 = self.act(self.deconv1(d2))

        # ====================================================
        # Final size correction
        # ====================================================

        d1 = F.interpolate(
            d1,
            size=(self.sensor_n, self.sensor_n),
            mode='nearest'
        )

        # (B,C,H,W) -> (B,H,W,C)

        d1 = d1.permute(0, 2, 3, 1)

        return d1


# ============================================================
# MSCRED API
# ============================================================

class MSCRED:

    def __init__(self, params):

        self.sensor_n = params[0]
        self.scale_n = params[1]
        self.step_max = params[2]

        self.device = torch.device("cpu")

        self.model = None

    # ========================================================
    # TRAIN
    # ========================================================

    def fit(self,
            X_train,
            Y_train,
            batch_size=32,
            epochs=10):

        self.model = _MSCREDNetwork(
            self.sensor_n,
            self.scale_n,
            self.step_max
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3
        )

        criterion = nn.MSELoss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.8,
            patience=6,
            min_lr=1e-6
        )

        X_t = torch.tensor(X_train, dtype=torch.float32)
        Y_t = torch.tensor(Y_train, dtype=torch.float32)

        dataset = TensorDataset(X_t, Y_t)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.model.train()

        for epoch in range(epochs):

            epoch_loss = 0.0

            for xb, yb in loader:

                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()

                out = self.model(xb)

                loss = criterion(out, yb)

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)

            scheduler.step(avg_loss)

            print(
                f"Epoch {epoch+1}/{epochs} "
                f"- loss: {avg_loss:.6f}"
            )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, data):

        self.model.eval()

        with torch.no_grad():

            X_t = torch.tensor(
                data,
                dtype=torch.float32
            ).to(self.device)

            out = self.model(X_t)

            return out.cpu().numpy()
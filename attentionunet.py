
import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.sigmoid(self.psi(g1 + x1))
        return x * psi


class UNetAttention(nn.Module):
    def __init__(self, in_channels=3, out_channels=23, init_features=64, dropout_rate=0.3):
        super(UNetAttention, self).__init__()
        features = init_features
        
        # Encoder Blocks
        self.encoder1 = self._block(in_channels, features, name="enc1", dropout_rate=dropout_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._block(features, features * 2, name="enc2", dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._block(features * 2, features * 4, name="enc3", dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._block(features * 4, features * 8, name="enc4", dropout_rate=dropout_rate)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder5 = self._block(features * 8, features * 16, name="enc5", dropout_rate=dropout_rate)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._block(features * 16, features * 32, name="bottleneck", dropout_rate=dropout_rate)

        # Decoder Blocks
        self.upconv5 = nn.ConvTranspose2d(features * 32, features * 16, kernel_size=2, stride=2)
        self.att5 = AttentionGate(F_g=features * 16, F_l=features * 16, F_int=features * 8)
        self.decoder5 = self._block(features * 32, features * 16, name="dec5", dropout_rate=dropout_rate)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=features * 8, F_l=features * 8, F_int=features * 4)
        self.decoder4 = self._block(features * 16, features * 8, name="dec4", dropout_rate=dropout_rate)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=features * 4, F_l=features * 4, F_int=features * 2)
        self.decoder3 = self._block(features * 8, features * 4, name="dec3", dropout_rate=dropout_rate)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=features * 2, F_l=features * 2, F_int=features)
        self.decoder2 = self._block(features * 4, features * 2, name="dec2", dropout_rate=dropout_rate)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1", dropout_rate=dropout_rate)

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))

        bottleneck = self.bottleneck(self.pool5(enc5))

        dec5 = self.upconv5(bottleneck)
        enc5 = self.att5(dec5, enc5)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)

        dec4 = self.upconv4(dec5)
        enc4 = self.att4(dec4, enc4)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        enc3 = self.att3(dec3, enc3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = self.att2(dec2, enc2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        #return torch.sigmoid(self.conv(dec1)) # Use with BCEWithLogitsLoss
        return self.conv(dec1)

    def _block(self, in_channels, features, name, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),  # Add dropout
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )

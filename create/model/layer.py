#!/usr/bin/env python
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, e_loss_weight=0.25, mu=0.01, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._e_loss_weight = e_loss_weight
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        self._mu = mu
        self._epsilon = epsilon

    def forward(self, inputs):
        distances = (torch.sum(inputs ** 2, dim=1, keepdim=True) + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(inputs, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(inputs.shape)
        encoding_indices = encoding_indices.reshape(inputs.shape[0], -1)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._e_loss_weight * e_latent_loss

        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * (1 - self._mu) + self._mu * torch.sum(encodings, 0)
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n)
            dw = torch.matmul(encodings.t(), inputs)
            self._ema_w = nn.Parameter(self._ema_w * (1 - self._mu) + self._mu * dw)
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, encoding_indices, perplexity

class Encoder1(nn.Module):
    def __init__(self, multi, channel1):
        super().__init__()
        self.multi = multi
        self.conv1 = torch.nn.Sequential(
            nn.Conv1d(4, out_channels=channel1 // 2, kernel_size=8, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5, stride=4),
            nn.Dropout(0.1),
        )
        self.layernorm1 = nn.LayerNorm(channel1 // 2)
        self.conv11 = torch.nn.Sequential(
            nn.Conv1d(1, out_channels=channel1 // 4, kernel_size=8, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5, stride=4),
            nn.Dropout(0.1),
        )
        self.layernorm11 = nn.LayerNorm(channel1 // 4)
        self.conv12 = torch.nn.Sequential(
            nn.Conv1d(1, out_channels=channel1 // 4, kernel_size=8, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5, stride=4),
            nn.Dropout(0.1),
        )
        self.layernorm12 = nn.LayerNorm(channel1 // 4)
        self.conv = torch.nn.Sequential(
            nn.Conv1d(channel1, out_channels=channel1, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.ids = {'seq':[],'open':[],'loop':[]}
        self.dict = {'seq':4,'open':1,'loop':1}
        last = 0
        for omic in self.multi:
            self.ids[omic] = [last, last+self.dict[omic]]
            last = last+self.dict[omic]

    def forward(self, x0):
        out = []
        if 'seq' in self.multi:
            i1, i2 = self.ids['seq']
            out1 = self.conv1(x0[:, i1:i2])
            out1 = self.layernorm1(out1.permute(0,2,1)).permute(0,2,1)
            out.append(out1)
        if 'open' in self.multi:
            i1, i2 = self.ids['open']
            out2 = self.conv11(x0[:, i1:i2])
            out2 = self.layernorm11(out2.permute(0,2,1)).permute(0,2,1)
            out.append(out2)
        if 'loop' in self.multi:
            i1, i2 = self.ids['loop']
            out3 = self.conv12(x0[:, i1:i2])
            out3 = self.layernorm12(out3.permute(0,2,1)).permute(0,2,1)
            out.append(out3)
        out = torch.cat(out, 1)
        out = self.conv(out)
        return out

class Encoder2(nn.Module):
    def __init__(self, channel1, channel2):
        super().__init__()
        self.conv2 = torch.nn.Sequential(
            nn.Conv1d(in_channels=channel1, out_channels=channel2, kernel_size=8, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.5),
        )

    def forward(self, out):
        out = self.conv2(out)
        return out

class Encoder3(nn.Module):
    def __init__(self, channel2, channel3):
        super().__init__()
        self.conv3 = torch.nn.Sequential(
            nn.Conv1d(in_channels=channel2, out_channels=channel3, kernel_size=8, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
        )

    def forward(self, out):
        out = self.conv3(out)
        return out

class Decoder3(nn.Module):
    def __init__(self, embed_dim, channel4):
        super().__init__()
        self.convt3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear'),
            nn.ConvTranspose1d(embed_dim, channel4, kernel_size=9, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, out):
        return self.convt3(out)

class Decoder2(nn.Module):
    def __init__(self, channel4, channel5):
        super().__init__()
        self.convt2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='linear'),
            nn.ConvTranspose1d(channel4, channel5, kernel_size=9, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, out):
        return self.convt2(out)

class Decoder1(nn.Module):
    def __init__(self, multi, channel5):
        super().__init__()
        self.multi = multi
        self.convt1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='linear'),
            nn.ConvTranspose1d(channel5, 4, kernel_size=9, stride=1),
            nn.BatchNorm1d(4),
            nn.Sigmoid(),
        )
        self.convt11 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='linear'),
            nn.ConvTranspose1d(channel5, 1, kernel_size=9, stride=1),
            nn.ReLU(inplace=True),
        )
        self.convt12 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='linear'),
            nn.ConvTranspose1d(channel5, 1, kernel_size=9, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, out):
        decs = []
        if 'seq' in self.multi:
            decs.append(self.convt1(out))
        if 'open' in self.multi:
            decs.append(self.convt11(out))
        if 'loop' in self.multi:
            decs.append(self.convt12(out))
        dec = torch.cat(decs, 1)
        return dec
    
class split_quant(nn.Module):
    def __init__(self, enc_channel=256, embed_dim=256, n_embed=200, split=16, ema=True, e_loss_weight=0.25, mu=0.01):
        super().__init__()
        self.part = embed_dim // split
        self.split = split
        self.quantize_conv3 = nn.Conv1d(enc_channel, embed_dim, kernel_size=1, stride=1)
        
        if ema:
            self.vq = VectorQuantizerEMA(num_embeddings=n_embed, embedding_dim=self.part, e_loss_weight=e_loss_weight, mu=mu)
        else:
            self.vq = VectorQuantizer(num_embeddings=n_embed, embedding_dim=self.part, e_loss_weight=e_loss_weight)

    def forward(self, enc3):
        z_e = self.quantize_conv3(enc3).permute(0, 2, 1)
        shape0, shape1, shape2 = z_e.shape[0], z_e.shape[1], z_e.shape[2]
        quant, latent_loss, index, perplexity = self.vq(z_e.reshape(-1, self.part))
        quant = quant.reshape(shape0, shape1, shape2).permute(0, 2, 1)
        return quant, latent_loss, index.reshape(shape0, shape1, -1), perplexity

class VQVAE(nn.Module):
    def __init__(self, multi=['seq','open','loop'], channel1=256, channel2=256, channel3=128, channel4=300, channel5=300, embed_dim=128, n_embed=200, split=16, ema=True, e_loss_weight=0.25, mu=0.01):
        super().__init__()
        self.multi = multi
        self.channel1, self.channel2, self.channel3 = channel1, channel2, channel3
        self.channel4, self.channel5 = channel4, channel5
        self.embed_dim, self.n_embed = embed_dim, n_embed

        self.enc1 = Encoder1(self.multi, self.channel1)
        self.enc2 = Encoder2(self.channel1, self.channel2)
        self.enc3 = Encoder3(self.channel2, self.channel3)

        self.quantize3 = split_quant(self.channel3, self.embed_dim, self.n_embed, split=split, ema=ema, e_loss_weight=e_loss_weight, mu=mu)

        self.dec3 = Decoder3(self.embed_dim, self.channel4)
        self.dec2 = Decoder2(self.channel4, self.channel5)
        self.dec1 = Decoder1(self.multi, self.channel5)

    def forward(self, input):
        enc1 = self.enc1(input)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        
        quant3, diff3, index3, perplexity3 = self.quantize3(enc3)

        dec3 = self.dec3(quant3)
        dec2 = self.dec2(dec3)
        dec = self.dec1(dec2)
        return quant3, dec, index3, diff3, perplexity3

# CREATE
class create(nn.Module):
    def __init__(self, num_class=5, multi=['seq','open','loop'], channel1=256, channel2=256, channel3=128, channel4=300, channel5=300, embed_dim=128, n_embed=200, split=16, ema=True, e_loss_weight=0.25, mu=0.01):
        super().__init__()
        self.vqvae = VQVAE(multi=multi, channel1=channel1, channel2=channel2, channel3=channel3, channel4=channel4, channel5=channel5, embed_dim=embed_dim, n_embed=n_embed, split=split, ema=ema, e_loss_weight=e_loss_weight, mu=mu)

        self.fc = torch.nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(26 * embed_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, num_class),
        )
        self.activation0 = nn.Sigmoid()

    def forward(self, x0):
        x3, out, index3, latent_loss, perplexity3 = self.vqvae(x0)
        x_0 = torch.flatten(x3.permute(0, 2, 1), 1)
        x = self.fc(x_0)
        x00 = self.activation0(x)
        return x, x00, out, index3, latent_loss, perplexity3

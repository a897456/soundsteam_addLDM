# Copyright (C) 2023 Katsuya Iida. All rights reserved.

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
try:
    import pytorch_lightning as pl
except ImportError:
    class pl:
        class LightningModule:
            pass

        class Callback:
            pass
from itertools import chain
import random
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from meldataset import mel_spectrogram

class ResNet1d(nn.Module):
    def __init__(
        self,
        n_channels,
        kernel_size: int = 7,
        padding: str = 'valid',
        dilation: int = 1
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self._padding_size = (kernel_size // 2) * dilation
        self.conv0 = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation)
        self.conv1 = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=1)

    def forward(self, input):
        y = input
        x = self.conv0(input)
        x = F.elu(x)
        x = self.conv1(x)
        if self.padding == 'valid':
            y = y[:, :, self._padding_size:-self._padding_size]
        x += y
        x = F.elu(x)
        return x


class ResNet2d(nn.Module):
    def __init__(
        self,
        n_channels: int,
        factor: int,
        stride: Tuple[int, int]
    ) -> None:
        # https://arxiv.org/pdf/2005.00341.pdf
        # The original paper uses layer normalization, but here
        # we use batch normalization.
        super().__init__()
        self.conv0 = nn.Conv2d(
            n_channels,
            n_channels,
            kernel_size=(3, 3),
            padding='same')
        self.bn0 = nn.BatchNorm2d(
            n_channels
        )
        self.conv1 = nn.Conv2d(
            n_channels,
            factor * n_channels,
            kernel_size=(stride[0] + 2, stride[1] + 2),
            stride=stride)
        self.bn1 = nn.BatchNorm2d(
            factor * n_channels
        )
        self.conv2 = nn.Conv2d(
            n_channels,
            factor * n_channels,
            kernel_size=1,
            stride=stride)
        self.bn2 = nn.BatchNorm2d(
            factor * n_channels
        )
        self.pad = nn.ReflectionPad2d([
            (stride[1] + 1) // 2,
            (stride[1] + 2) // 2,
            (stride[0] + 1) // 2,
            (stride[0] + 2) // 2,
        ])
        self.activation = nn.LeakyReLU(0.3)

    def forward(self, input):
        x = self.conv0(input)
        x = self.bn0(x)
        x = self.activation(x)
        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn1(x)

        # shortcut
        y = self.conv2(input)
        y = self.bn2(y)

        x += y
        x = self.activation(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        padding: str,
        stride: int
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            ResNet1d(n_channels // 2, padding=padding, dilation=1),
            ResNet1d(n_channels // 2, padding=padding, dilation=3),
            ResNet1d(n_channels // 2, padding=padding, dilation=9),
            nn.Conv1d(
                n_channels // 2, n_channels,
                kernel_size=2 * stride,
                padding=(2 * stride) // 2 if padding == 'same' else 0,
                stride=stride),
            nn.ELU(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        padding: str,
        stride: int
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(
                n_channels, n_channels // 2,
                kernel_size=2 * stride,
                padding=(2 * stride) // 2 if padding == 'same' else 0,
                stride=stride),
            nn.ELU(),
            ResNet1d(n_channels // 2, padding=padding, dilation=1),
            ResNet1d(n_channels // 2, padding=padding, dilation=3),
            ResNet1d(n_channels // 2, padding=padding, dilation=9),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class Encoder(nn.Module):
    def __init__(self, n_channels: int, padding):
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            nn.Conv1d(1, n_channels, kernel_size=7, padding=padding),
            nn.ELU(),
            EncoderBlock(2 * n_channels, padding=padding, stride=2),
            EncoderBlock(4 * n_channels, padding=padding, stride=4),
            EncoderBlock(8 * n_channels, padding=padding, stride=5),
            EncoderBlock(16 * n_channels, padding=padding, stride=8),
            nn.Conv1d(16 * n_channels, 16 * n_channels, kernel_size=3, padding=padding),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class Decoder(nn.Module):
    def __init__(self, n_channels: int, padding):
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            nn.Conv1d(16 * n_channels, 16 * n_channels, kernel_size=7, padding=padding),
            nn.ELU(),
            DecoderBlock(16 * n_channels, padding=padding, stride=8),
            DecoderBlock(8 * n_channels, padding=padding, stride=5),
            DecoderBlock(4 * n_channels, padding=padding, stride=4),
            DecoderBlock(2 * n_channels, padding=padding, stride=2),
            nn.Conv1d(n_channels, 1, kernel_size=7, padding=padding),
            nn.Tanh(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class ResidualVectorQuantizer(nn.Module):
    weight: torch.Tensor
    running_mean: torch.Tensor
    code_count: torch.Tensor

    def __init__(
        self,
        num_quantizers: int,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        code_replace_threshold: float = 0.0001,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.register_buffer("weight", torch.empty(num_quantizers, num_embeddings, embedding_dim))
        self.register_buffer("running_mean", torch.empty(num_quantizers, num_embeddings, embedding_dim))
        self.register_buffer("code_count", torch.empty(num_quantizers, num_embeddings))
        self.decay = decay
        self.eps = eps
        self.code_replace_threshold = code_replace_threshold
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.uniform_(self.weight)
        self.running_mean[:] = self.weight
        init.ones_(self.code_count)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # input: [..., chennel]
        if self.training:
            # Enabling bitrate scalability with quantizer dropout
            n = random.randrange(1, self.num_quantizers)
        else:
            n = self.num_quantizers
        codes = []
        r = input.type_as(self.running_mean).detach()
        with torch.no_grad():
            for i in range(n):
                w = self.weight[i]
                # r: [..., num_embeddings]
                dist = torch.cdist(r, w)
                k = torch.argmin(dist, axis=-1)
                codes.append(k)
                self._update_averages(i, r, k)
                r = r - F.embedding(k, w)
        quantized = input - r
        commitment_loss = torch.mean(torch.square(input - quantized.detach()))
        self.weight.data[:] = self.running_mean / torch.unsqueeze(self.eps + self.code_count, axis=-1)
        return quantized, torch.stack(codes, input.ndim - 1), commitment_loss

    def dequantize(self, input: torch.Tensor, n: Optional[int] = None) -> torch.Tensor:
        # input: [batch_size, length, num_quantizers]
        if n is None:
            n = input.shape[-1]
        assert 0 < n <= self.num_quantizers
        res = 0
        with torch.no_grad():
            for i in range(n):
                k = input[:, :, i]
                w = self.weight[i]
                res += F.embedding(k, w)
        return res

    def _update_averages(self, i: int, r: torch.Tensor, k: torch.Tensor) -> None:
        # https://arxiv.org/pdf/1906.00446.pdf
        # Generating Diverse High-Fidelity Images with VQ-VAE-2
        # 2.1 Vector Quantized Variational AutoEncode

        # k: [...]
        one_hot_k = F.one_hot(torch.flatten(k), self.num_embeddings).type_as(self.code_count)
        code_count_update = torch.mean(one_hot_k, axis=0)
        self.code_count[i].lerp_(code_count_update, 1 - self.decay)

        # r: [..., embedding_dim]
        r = r.reshape(-1, self.embedding_dim)
        running_mean_update = (one_hot_k.T @ r) / r.shape[0]
        self.running_mean[i].lerp_(running_mean_update, 1 - self.decay)

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def replace_vectors(self) -> int:
        # https://arxiv.org/pdf/2107.03312.pdf
        # C. Residual Vector Quantizer:

        # The original paper replaces with an input frame randomly
        # sampled within the current batch.
        # Here we replace with random average of running mean instead.
        num_replaced = torch.sum(self.code_count < self.code_replace_threshold).item()
        if num_replaced > 0:
            for i in range(self.num_quantizers):
                mask = self.code_count[i] < self.code_replace_threshold
                # mask: [num_quantizers, num_embeddings]
                w = torch.rand_like(self.code_count[i])
                w /= torch.sum(w)
                self.running_mean[i, mask] = w.type_as(self.running_mean) @ self.running_mean[i]
                self.code_count[i, mask] = w.type_as(self.code_count) @ self.code_count[i]

        return num_replaced

    @torch.no_grad()
    def calc_entropy(self) -> float:
        p = self.code_count / (self.eps + torch.sum(self.code_count, axis=-1, keepdim=True))
        return -torch.sum(torch.log(p) * p).item() / self.num_quantizers


class WaveDiscriminator(nn.Module):
    r"""MelGAN discriminator from https://arxiv.org/pdf/1910.06711.pdf
    """
    def __init__(self, resolution: int = 1, n_channels: int = 4) -> None:
        super().__init__()
        assert resolution >= 1
        if resolution == 1:
            self.avg_pool = nn.Identity()
        else:
            self.avg_pool = nn.AvgPool1d(resolution * 2, stride=resolution)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.layers = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(1, n_channels, kernel_size=15, padding=7)),
            nn.utils.weight_norm(nn.Conv1d(n_channels, 4 * n_channels, kernel_size=41, stride=4, padding=20, groups=4)),
            nn.utils.weight_norm(nn.Conv1d(4 * n_channels, 16 * n_channels, kernel_size=41, stride=4, padding=20, groups=16)),
            nn.utils.weight_norm(nn.Conv1d(16 * n_channels, 64 * n_channels, kernel_size=41, stride=4, padding=20, groups=64)),
            nn.utils.weight_norm(nn.Conv1d(64 * n_channels, 256 * n_channels, kernel_size=41, stride=4, padding=20, groups=256)),
            nn.utils.weight_norm(nn.Conv1d(256 * n_channels, 256 * n_channels, kernel_size=5, padding=2)),
            nn.utils.weight_norm(nn.Conv1d(256 * n_channels, 1, kernel_size=3, padding=1)),
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.avg_pool(x)
        feats = []
        for layer in self.layers[:-1]:
            x = layer(x)
            feats.append(x)
            x = self.activation(x)
        feats.append(self.layers[-1](x))
        return feats


class STFTDiscriminator(nn.Module):
    r"""STFT-based discriminator from https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(
        self, n_fft: int = 1024, hop_length: int = 256,
        n_channels: int = 32
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        n = n_fft // 2 + 1
        for _ in range(6):
            n = (n - 1) // 2 + 1
        self.layers = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=7, padding='same'),
            nn.LeakyReLU(0.3, inplace=True),
            ResNet2d(n_channels, 2, stride=(2, 1)),
            ResNet2d(2 * n_channels, 2, stride=(2, 2)),
            ResNet2d(4 * n_channels, 1, stride=(2, 1)),
            ResNet2d(4 * n_channels, 2, stride=(2, 2)),
            ResNet2d(8 * n_channels, 1, stride=(2, 1)),
            ResNet2d(8 * n_channels, 2, stride=(2, 2)),
            nn.Conv2d(16 * n_channels, 1, kernel_size=(n, 1))
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[1] == 1
        # input: [batch, channel, sequence]
        x = torch.squeeze(input, 1).to(torch.float32)  # torch.stft() doesn't accept float16
        x = torch.stft(x, self.n_fft, self.hop_length, normalized=True, onesided=True, return_complex=True)
        x = torch.abs(x)
        x = torch.unsqueeze(x, dim=1)
        x = self.layers(x)
        return x


class ReconstructionLoss(nn.Module):
    """Reconstruction loss from https://arxiv.org/pdf/2107.03312.pdf
    but uses STFT instead of mel-spectrogram
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        loss = 0
        input = input.to(torch.float32)
        target = target.to(torch.float32)
        for i in range(6, 12):
            s = 2 ** i
            alpha = (s / 2) ** 0.5
            # We use STFT instead of 64-bin mel-spectrogram as n_fft=64 is too small
            # for 64 bins.
            x = torch.stft(input, n_fft=s, hop_length=s // 4, win_length=s, normalized=True, onesided=True, return_complex=True)
            x = torch.abs(x)
            y = torch.stft(target, n_fft=s, hop_length=s // 4, win_length=s, normalized=True, onesided=True, return_complex=True)
            y = torch.abs(y)
            if x.shape[-1] > y.shape[-1]:
                x = x[:, :, :y.shape[-1]]
            elif x.shape[-1] < y.shape[-1]:
                y = y[:, :, :x.shape[-1]]
            loss += torch.mean(torch.abs(x - y))
            loss += alpha * torch.mean(torch.square(torch.log(x + self.eps) - torch.log(y + self.eps)))
        return loss / (12 - 6)


class ReconstructionLoss2(nn.Module):
    """Reconstruction loss from https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, sample_rate, eps=1e-5):
        super().__init__()
        import torchaudio
        self.layers = nn.ModuleList()
        self.alpha = []
        self.eps = eps
        for i in range(6, 12):
            melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=int(2 ** i),
                win_length=int(2 ** i),
                hop_length=int(2 ** i / 4),
                n_mels=64)
            self.layers.append(melspec)
            self.alpha.append((2 ** i / 2) ** 0.5)

    def forward(self, input, target):
        loss = 0
        for alpha, melspec in zip(self.alpha, self.layers):
            x = melspec(input)
            y = melspec(target)
            if x.shape[-1] > y.shape[-1]:
                x = x[:, y.shape[-1]]
            elif x.shape[-1] < y.shape[-1]:
                y = y[:, x.shape[-1]]
            loss += torch.mean(torch.abs(x - y))
            loss += alpha * torch.mean(torch.square(torch.log(x + self.eps) - torch.log(y + self.eps)))
        return loss


class StreamableModel(pl.LightningModule):
    def __init__(
        self,
        n_channels: int = 32,
        num_quantizers: int = 8,
        num_embeddings: int = 4096,
        padding: str = "valid",
        batch_size: int = 32,
        sample_rate: int = 24_000,
        segment_length: int = 32270,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.9,
        dataset: str = 'librispeech'
    ) -> None:
        # https://arxiv.org/pdf/2009.02095.pdf
        # 2. Method
        # SEANet uses Adam with lr=1e-4, beta1=0.5, beta2=0.9
        # batch_size=16
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.encoder = Encoder(n_channels, padding)
        self.decoder = Decoder(n_channels, padding)
        self.quantizer = ResidualVectorQuantizer(
            num_quantizers, num_embeddings, n_channels * 16)

        # self.wave_discriminators = nn.ModuleList([
        #     WaveDiscriminator(resolution=1),
        #     WaveDiscriminator(resolution=2),
        #     WaveDiscriminator(resolution=4)
        # ])
        self.rec_loss = ReconstructionLoss()
        self.stft_discriminator = STFTDiscriminator()
        # self.msd = MultiScaleDiscriminator()  # 张怀峰添加
        # self.mpd = MultiPeriodDiscriminator()  # 张怀峰添加
        self.lr=lr
        self.b1=b1
        self.b2 = b2
    def configure_optimizers(self):
        # lr = self.hparams.lr
        # b1 = self.hparams.b1
        # b2 = self.hparams.b2
        optimizer_g = torch.optim.Adam(
            chain(self.encoder.parameters(),self.decoder.parameters()),
            lr=self.lr, betas=(self.b1, self.b2))
        optimizer_d = torch.optim.Adam(
            chain(msd.parameters(),mpd.parameters()),
            lr=self.lr, betas=(self.b1, self.b2))
        return [optimizer_g, optimizer_d], []

    def forward(self, input):
        x = self.encoder(input)
        x = torch.transpose(x, -1, -2)
        x, codes, codebook_loss = self.quantizer(x)
        x = torch.transpose(x, -1, -2)
        x = self.decoder(x)
        return x, codes, codebook_loss

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()
        input = batch[:, None, :]
        # input: [batch, channel, sequence]


        # discriminator
        self.toggle_optimizer(optimizer_d)
        output, _, q_loss = self.forward(input)
        optimizer_d.zero_grad()

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = mpd(input, output.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = msd(input, output.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        loss_disc_all = loss_disc_s + loss_disc_f

        self.manual_backward(loss_disc_all)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)


        # Generator
        self.toggle_optimizer(optimizer_g)
        output, _, q_loss = self.forward(input)
        optimizer_g.zero_grad()

        # L1 Mel-Spectrogram Loss
        y_mel = mel_spectrogram(input.squeeze(1), 1024, 80, 8000, 256, 1024, 0, 8000)
        y_g_hat_mel = mel_spectrogram(output.squeeze(1), 1024, 80, 8000, 256, 1024, 0, 8000)
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(input, output)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(input, output)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        self.manual_backward(loss_gen_all)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        self.log("g_mel_loss", loss_mel,prog_bar=True)
        self.log("g_fm_msd_loss", loss_fm_s,prog_bar=True)
        self.log("g_fm_mpd_loss", loss_gen_f,prog_bar=True)
        self.log("g_loss", loss_gen_all, prog_bar=True)
        self.log("g_gen_msd_loss", loss_gen_s, prog_bar=True)
        self.log("g_gen_mpd_loss", loss_gen_f, prog_bar=True)
        self.log("d_loss", loss_disc_all, prog_bar=True)
        self.log("d_msd_loss", loss_disc_s, prog_bar=True)
        self.log("d_mpd_loss", loss_disc_f, prog_bar=True)

        codes_entropy = self.quantizer.calc_entropy()
        self.log("codes_entropy", codes_entropy, prog_bar=True)

        num_replaced = self.quantizer.replace_vectors()
        self.log("num_replaced", float(num_replaced), prog_bar=True)


    def train_dataloader(self):
        return self._make_dataloader(True)

    def _make_dataloader(self, train: bool):
        import torchaudio

        def collate(examples):
            return torch.stack(examples)

        class VoiceDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, sample_rate, segment_length):
                self._dataset = dataset
                self._sample_rate = sample_rate
                self._segment_length = segment_length

            def __getitem__(self, index):
                import random
                x, sample_rate, *_ = self._dataset[index]
                x = torchaudio.functional.resample(x, sample_rate, self._sample_rate)
                assert x.shape[0] == 1
                x = torch.squeeze(x)
                x *= 0.95 / torch.max(x)
                assert x.dim() == 1
                if x.shape[0] < self._segment_length:
                    x = F.pad(x, [0, self._segment_length - x.shape[0]], "constant")
                pos = random.randint(0, x.shape[0] - self._segment_length)
                x = x[pos:pos + self._segment_length]
                return x

            def __len__(self):
                return len(self._dataset)

        if self.hparams.dataset == 'yesno':
            ds = torchaudio.datasets.YESNO("./data", download=True)
        elif self.hparams.dataset == 'librispeech-dev':
            ds = torchaudio.datasets.LIBRISPEECH("./data", url="dev-clean")
        elif self.hparams.dataset == 'librispeech':
            url = "train-clean-100" if train else "dev-clean"
            ds = torchaudio.datasets.LIBRISPEECH("./data", url=url)
        else:
            raise ValueError()
        ds = VoiceDataset(ds, self.hparams.sample_rate, self.hparams.segment_length)

        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.hparams['batch_size'], shuffle=True,
            collate_fn=collate)
        return loader


class KMeanCodebookInitCallback(pl.Callback):
    def on_fit_start(self, trainer, model):
        # https://arxiv.org/pdf/2107.03312.pdf
        # C. Residual Vector Quantizer
        # run the k-means
        # algorithm on the first training batch and use the learned
        # centroids as initialization
        batch = next(iter(model.train_dataloader()))
        input = batch[:, None, :].to(model.device)
        with torch.no_grad():
            x=model.encoder(input)
            x=torch.flatten(x)
            # x = torch.flatten(model.encoder(input))
            mean = torch.mean(x, axis=0)
            std = torch.std(x, axis=0)
            torch.nn.init.normal_(model.quantizer.weight, mean=mean, std=std)
        print(f"KMeanCodebookInitCallback {mean} {std}")


def train():
    mpd.train()
    msd.train()
    model = StreamableModel(
        batch_size=32,
        sample_rate=8_000,
        segment_length=32270,
        padding='same',
        dataset='librispeech')
    trainer = pl.Trainer(
        max_epochs=151,
        log_every_n_steps=2,
        precision='16-mixed',
        # logger=pl.loggers.CSVLogger("."),
        logger=pl.loggers.TensorBoardLogger("lightning_logs", name="20240225_8kHz_8x4096_2548_addMSDMPD"),
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_last=True,
                                         # every_n_train_steps=1500,
                                         save_top_k=-1),
            KMeanCodebookInitCallback(),
        ],
    )
    trainer.fit(
        # model,None, None, None, 'E:/000/soundstream-pytorch-main/lightning_logs/20240224_8kHz_8x4096_2548_addMPD/version_3/checkpoints/epoch=56-step=100500.ckpt'
        model,None,None,None,None
        # model,None, None, None, 'E:/000/soundstream-pytorch-main/lightning_logs/20240127_8kHz_8x4096_2548/version_1/checkpoints/epoch=143-step=256500.ckpt'
    )

    return model

#
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    train()
#
# import torchaudio
# import os
# from scipy.io.wavfile import read,write
#
# if __name__ == "__main__":
#     model=StreamableModel.load_from_checkpoint('E:/000/soundstream-pytorch-main/lightning_logs/20240225_8kHz_8x4096_2548_addMPD/version_1/checkpoints/epoch=103-step=184500.ckpt')
#
#     model.eval()
#     testfile_path='C:/Users/admin/Desktop/LibriSpeech_8k_testfile'
#     testfile_list = os.listdir(testfile_path)
#     model.eval()
#     model.remove_weight_norm()
#     with torch.no_grad():
#         for i, testfile_name in enumerate(testfile_list):
#             x, sr = torchaudio.load(testfile_name)
#             x = torch.FloatTensor(x)
#             x = x.to('cuda')
#             y_hat = model(x)
#             y_hat = y_hat[0]
#             y_hat = y_hat.squeeze()
#             y_hat = y_hat.cpu()
#             y_hat = y_hat.detach().numpy()
#             file_out_path = "C:/Users/admin/Desktop/testfile_baseline_2k4_epoch"
#             output_file = os.path.join(file_out_path, 'output_file.wav')
#             write(output_file, sr, y_hat)


# import torchaudio
# import os
# from scipy.io.wavfile import read,write
#
# if __name__ == "__main__":
#     model=StreamableModel.load_from_checkpoint('E:/000/soundstream-pytorch-main/lightning_logs/20240225_8kHz_8x4096_2548_addMPD/version_1/checkpoints/epoch=99-step=177000.ckpt')
#     testfile_path_input = './LibriSpeech_8k_testfile'
#     testfile_path_output = './testfile_addMSDMPD3_2k4_2'
#     testfile_list = os.listdir(testfile_path_input)
#     model.eval()
#     # model.remove_weight_norm()
#     with torch.no_grad():
#         for i, testfile_name in enumerate(testfile_list):
#             x, sr = torchaudio.load(os.path.join(testfile_path_input, testfile_name))
#             # x ,sr= np.load(os.path.join(testfile_path_input,testfile_name))
#             x = torch.FloatTensor(x)
#             x = x.to('cuda')
#             y_hat = model(x)
#             y_hat = y_hat[0]
#             y_hat = y_hat.squeeze()
#             y_hat = y_hat.cpu()
#             y_hat = y_hat.detach().numpy()
#             output_file = os.path.join(testfile_path_output, os.path.splitext(testfile_name)[0] + '.wav')
#             write(output_file, sr, y_hat)
#             print(output_file)
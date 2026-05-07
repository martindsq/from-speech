import torchaudio
from torch import Tensor, nn, empty, full, randn_like
from torch.nn import functional as F
from .utils import AUDIO_SAMPLE_RATE

class RandomAlign(nn.Module):
    """Align a waveform inside a one second window.

    Parameters
    ----------
    stride: float or tuple[float, float]
        Alignment value or range between 0 and 1.

        If a float:
            - 0.0 places the waveform at the far left.
            - 0.5 centers the waveform.
            - 1.0 places the waveform at the far right.

        If a tuple `(min, max)`:
            a stride is sampled uniformly from `[min, max]` each time.

        If the waveform is longer than one second, the same value controls
        the crop position.
    """

    def __init__(self, stride: float | tuple = (0.0, 1.0)) -> None:
        super().__init__()
        self.stride = self._setup_stride(stride)
        self.sample_rate = AUDIO_SAMPLE_RATE

    @staticmethod
    def _setup_stride(stride: float | tuple) -> tuple:
        if isinstance(stride, float):
            return (stride, stride)
        if isinstance(stride, tuple) and len(stride) == 2:
            return stride
        raise TypeError("stride should be a float or a tuple of two floats")

    @staticmethod
    def get_params(stride: tuple[float, float]) -> float:
        """Get the stride parameter for the transform."""
        min_stride, max_stride = stride
        if min_stride == max_stride:
            return min_stride
        return float(empty(1).uniform_(min_stride, max_stride).item())

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Parameters
        ----------
        waveform: Tensor
            Tensor of shape [T] or [B, T].

        Returns
        -------
        out: Tensor
            Tensor of shape [B, sample_rate].
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        stride = self.get_params(self.stride)

        B, T = waveform.shape

        if T == self.sample_rate:
            return waveform

        if T < self.sample_rate:
            total_pad = self.sample_rate - T
            left_pad = int(round(total_pad * stride))
            right_pad = total_pad - left_pad

            return F.pad(waveform, (left_pad, right_pad))

        total_crop = T - self.sample_rate
        start = int(round(total_crop * stride))
        end = start + self.sample_rate

        return waveform[..., start:end]

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__} (stride={self.stride})")


class RandomNoise(nn.Module):
    """Add white noise to a waveform at a random signal-to-noise ratio.

    Parameters
    ----------
    noise: float or tuple[float, float]
        Signal-to-noise ratio in decibels.

        If a float:
            the same SNR is used every time.

        If a tuple `(min, max)`:
            an SNR is sampled uniformly from `[min, max]` each time.
    """

    def __init__(self, noise: float | tuple = (10.0, 30.0)) -> None:
        super().__init__()
        self.snr_db = self._setup_snr_db(noise)

    @staticmethod
    def _setup_snr_db(snr_db: float | tuple) -> tuple:
        if isinstance(snr_db, (float, int)):
            snr_db = float(snr_db)
            return (snr_db, snr_db)
        if isinstance(snr_db, tuple) and len(snr_db) == 2:
            min_snr = float(snr_db[0])
            max_snr = float(snr_db[1])
            if min_snr > max_snr:
                raise ValueError("snr_db min must be <= max")
            return (min_snr, max_snr)
        raise TypeError("snr_db should be a float or a tuple of two floats")

    @staticmethod
    def get_params(snr_db: tuple[float, float]) -> float:
        """Get the SNR parameter for the transform."""
        min_snr, max_snr = snr_db
        if min_snr == max_snr:
            return min_snr
        return float(empty(1).uniform_(min_snr, max_snr).item())

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Parameters
        ----------
        waveform: Tensor
            Tensor of shape [T] or [B, T].

        Returns
        -------
        out: Tensor
            Tensor with the shape [B, T].
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        noise = randn_like(waveform)
        snr_db = self.get_params(self.snr_db)
        snr = full(
            waveform.shape[:-1],
            snr_db,
            dtype=waveform.dtype,
            device=waveform.device,
        )
        out = torchaudio.functional.add_noise(waveform, noise, snr)
        return out

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__} (snr_db={self.snr_db})")


class RandomPitch(nn.Module):
    """Shift waveform pitch by a random number of semitones.

    Parameters
    ----------
    pitch: float or tuple[float, float]
        Pitch shift in semitones.

        If a float:
            the same shift is used every time.

        If a tuple `(min, max)`:
            a shift is sampled uniformly from `[min, max]` each time.
    """

    def __init__(self, pitch: float | tuple = (-2.0, 2.0)) -> None:
        super().__init__()
        self.n_steps = self._setup_n_steps(pitch)
        self.sample_rate = AUDIO_SAMPLE_RATE

    @staticmethod
    def _setup_n_steps(n_steps: float | tuple) -> tuple:
        if isinstance(n_steps, (float, int)):
            n_steps = float(n_steps)
            return (n_steps, n_steps)
        if isinstance(n_steps, tuple) and len(n_steps) == 2:
            min_steps = float(n_steps[0])
            max_steps = float(n_steps[1])
            if min_steps > max_steps:
                raise ValueError("n_steps min must be <= max")
            return (min_steps, max_steps)
        raise TypeError("n_steps should be a float or a tuple of two floats")

    @staticmethod
    def get_params(n_steps: tuple[float, float]) -> float:
        """Get the pitch-shift parameter for the transform."""
        min_steps, max_steps = n_steps
        if min_steps == max_steps:
            return min_steps
        return float(empty(1).uniform_(min_steps, max_steps).item())

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Parameters
        ----------
        waveform: Tensor
            Tensor of shape [T] or [B, T].

        Returns
        -------
        out: Tensor
            Tensor with shape [B, T].
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        n_steps = self.get_params(self.n_steps)
        out = torchaudio.functional.pitch_shift(
            waveform,
            sample_rate=self.sample_rate,
            n_steps=n_steps,
        )
        return out

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__} (n_steps={self.n_steps})")


class RandomNotch(nn.Module):
    """Apply a notch filter at a random center frequency.

    Parameters
    ----------
    center: float or tuple[float, float]
        Center frequency in Hz.

        If a float:
            the same center frequency is used every time.

        If a tuple `(min, max)`:
            a center frequency is sampled uniformly from `[min, max]` each time.
    Q: float or tuple[float, float]
        Quality factor of the notch filter.

        If a float:
            the same Q is used every time.

        If a tuple `(min, max)`:
            a Q is sampled uniformly from `[min, max]` each time.
    """

    def __init__(self, center: float | tuple = (120.0, 4000.0), Q: float | tuple = (6.0, 10.0)) -> None:
        super().__init__()
        self.center = self._setup_center(center)
        self.Q = self._setup_Q(Q)
        self.sample_rate = AUDIO_SAMPLE_RATE

    @staticmethod
    def _setup_center(center: float | tuple) -> tuple:
        if isinstance(center, (float, int)):
            center = float(center)
            return (center, center)
        if isinstance(center, tuple) and len(center) == 2:
            min_center = float(center[0])
            max_center = float(center[1])
            if min_center > max_center:
                raise ValueError("center min must be <= max")
            return (min_center, max_center)
        raise TypeError("center should be a float or a tuple of two floats")

    @staticmethod
    def _setup_Q(Q: float | tuple) -> tuple:
        if isinstance(Q, (float, int)):
            Q = float(Q)
            return (Q, Q)
        if isinstance(Q, tuple) and len(Q) == 2:
            min_Q = float(Q[0])
            max_Q = float(Q[1])
            if min_Q > max_Q:
                raise ValueError("Q min must be <= max")
            return (min_Q, max_Q)
        raise TypeError("Q should be a float or a tuple of two floats")

    @staticmethod
    def get_params(center: tuple, Q: tuple) -> tuple:
        """Get the center frequency and Q parameters for the transform."""
        min_center, max_center = center
        if min_center == max_center:
            center = min_center
        else:
            center = float(empty(1).uniform_(min_center, max_center).item())
        min_Q, max_Q = Q
        if min_Q == max_Q:
            Q = min_Q
        else:
            Q = float(empty(1).uniform_(min_Q, max_Q).item())
        return center, Q

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Parameters
        ----------
        waveform: Tensor
            Tensor of shape [T] or [B, T].

        Returns
        -------
        out: Tensor
            Tensor with shape [B, T].
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        center, Q = self.get_params(self.center, self.Q)
        out = torchaudio.functional.bandreject_biquad(
            waveform,
            self.sample_rate,
            central_freq=center,
            Q=Q
        )
        return out

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__} (center={self.center}, Q={self.Q})")


class RandomBandpass(nn.Module):
    """Apply a bandpass filter at a random center frequency.

    Parameters
    ----------
    cutoff: float or tuple[float, float]
        Center frequency in Hz.

        If a float:
            the same center frequency is used every time.

        If a tuple `(min, max)`:
            a center frequency is sampled uniformly from `[min, max]` each time.
    Q: float or tuple[float, float]
        Quality factor of the bandpass filter.
    """

    def __init__(self, cutoff: float | tuple = (500.0, 3000.0), Q: float | tuple = 0.707) -> None:
        super().__init__()
        self.cutoff = self._setup_cutoff(cutoff)
        self.Q = self._setup_Q(Q)
        self.sample_rate = AUDIO_SAMPLE_RATE

    @staticmethod
    def _setup_cutoff(cutoff: float | tuple) -> tuple:
        if isinstance(cutoff, (float, int)):
            cutoff = float(cutoff)
            return (cutoff, cutoff)
        if isinstance(cutoff, tuple) and len(cutoff) == 2:
            min_cutoff = float(cutoff[0])
            max_cutoff = float(cutoff[1])
            if min_cutoff > max_cutoff:
                raise ValueError("cutoff min must be <= max")
            return (min_cutoff, max_cutoff)
        raise TypeError("cutoff should be a float or a tuple of two floats")

    @staticmethod
    def _setup_Q(Q: float | tuple) -> tuple:
        if isinstance(Q, (float, int)):
            Q = float(Q)
            return (Q, Q)
        if isinstance(Q, tuple) and len(Q) == 2:
            min_Q = float(Q[0])
            max_Q = float(Q[1])
            if min_Q > max_Q:
                raise ValueError("Q min must be <= max")
            return (min_Q, max_Q)
        raise TypeError("Q should be a float or a tuple of two floats")

    @staticmethod
    def get_params(cutoff: tuple[float, float], Q: tuple[float, float]) -> tuple[float, float]:
        """Get the cutoff frequency and Q parameters for the transform."""
        min_cutoff, max_cutoff = cutoff
        if min_cutoff == max_cutoff:
            cutoff = min_cutoff
        else:
            cutoff = float(empty(1).uniform_(min_cutoff, max_cutoff).item())
        min_Q, max_Q = Q
        if min_Q == max_Q:
            Q = min_Q
        else:
            Q = float(empty(1).uniform_(min_Q, max_Q).item())
        return cutoff, Q

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Parameters
        ----------
        waveform: Tensor
            Tensor of shape [T] or [B, T].

        Returns
        -------
        out: Tensor
            Tensor with shape [B, T].
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        cutoff, Q = self.get_params(self.cutoff, self.Q)
        out = torchaudio.functional.bandpass_biquad(
            waveform,
            self.sample_rate,
            central_freq=cutoff,
            Q=Q,
        )
        return out

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__} (cutoff={self.cutoff}, Q={self.Q})")


class RandomLowpass(nn.Module):
    """Apply a lowpass filter at a random cutoff frequency.

    Parameters
    ----------
    cutoff: float or tuple[float, float]
        Cutoff frequency in Hz.

        If a float:
            the same cutoff frequency is used every time.

        If a tuple `(min, max)`:
            a cutoff frequency is sampled uniformly from `[min, max]` each time.
    Q: float or tuple[float, float]
        Quality factor of the lowpass filter.
    """

    def __init__(self, cutoff: float | tuple = (2500.0, 6000.0), Q: float | tuple = 0.707) -> None:
        super().__init__()
        self.cutoff = self._setup_cutoff(cutoff)
        self.Q = self._setup_Q(Q)
        self.sample_rate = AUDIO_SAMPLE_RATE

    @staticmethod
    def _setup_cutoff(cutoff: float | tuple) -> tuple:
        if isinstance(cutoff, (float, int)):
            cutoff = float(cutoff)
            return (cutoff, cutoff)
        if isinstance(cutoff, tuple) and len(cutoff) == 2:
            min_cutoff = float(cutoff[0])
            max_cutoff = float(cutoff[1])
            if min_cutoff > max_cutoff:
                raise ValueError("cutoff min must be <= max")
            return (min_cutoff, max_cutoff)
        raise TypeError("cutoff should be a float or a tuple of two floats")

    @staticmethod
    def _setup_Q(Q: float | tuple) -> tuple:
        if isinstance(Q, (float, int)):
            Q = float(Q)
            return (Q, Q)
        if isinstance(Q, tuple) and len(Q) == 2:
            min_Q = float(Q[0])
            max_Q = float(Q[1])
            if min_Q > max_Q:
                raise ValueError("Q min must be <= max")
            return (min_Q, max_Q)
        raise TypeError("Q should be a float or a tuple of two floats")

    @staticmethod
    def get_params(cutoff: tuple[float, float], Q: tuple[float, float]) -> tuple[float, float]:
        """Get the cutoff frequency and Q parameters for the transform."""
        min_cutoff, max_cutoff = cutoff
        if min_cutoff == max_cutoff:
            cutoff = min_cutoff
        else:
            cutoff = float(empty(1).uniform_(min_cutoff, max_cutoff).item())
        min_Q, max_Q = Q
        if min_Q == max_Q:
            Q = min_Q
        else:
            Q = float(empty(1).uniform_(min_Q, max_Q).item())
        return cutoff, Q

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Parameters
        ----------
        waveform: Tensor
            Tensor of shape [T] or [B, T].

        Returns
        -------
        out: Tensor
            Tensor with shape [B, T].
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        cutoff, Q = self.get_params(self.cutoff, self.Q)
        out = torchaudio.functional.lowpass_biquad(
            waveform,
            self.sample_rate,
            cutoff_freq=cutoff,
            Q=Q,
        )
        return out

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__} (cutoff={self.cutoff}, Q={self.Q})")


class RandomHighpass(nn.Module):
    """Apply a highpass filter at a random cutoff frequency.

    Parameters
    ----------
    cutoff: float or tuple[float, float]
        Cutoff frequency in Hz.

        If a float:
            the same cutoff frequency is used every time.

        If a tuple `(min, max)`:
            a cutoff frequency is sampled uniformly from `[min, max]` each time.
    Q: float or tuple[float, float]
        Quality factor of the highpass filter.
    """

    def __init__(self, cutoff: float | tuple = (80.0, 500.0), Q: float | tuple = 0.707) -> None:
        super().__init__()
        self.cutoff = self._setup_cutoff(cutoff)
        self.Q = self._setup_Q(Q)
        self.sample_rate = AUDIO_SAMPLE_RATE

    @staticmethod
    def _setup_cutoff(cutoff: float | tuple) -> tuple:
        if isinstance(cutoff, (float, int)):
            cutoff = float(cutoff)
            return (cutoff, cutoff)
        if isinstance(cutoff, tuple) and len(cutoff) == 2:
            min_cutoff = float(cutoff[0])
            max_cutoff = float(cutoff[1])
            if min_cutoff > max_cutoff:
                raise ValueError("cutoff min must be <= max")
            return (min_cutoff, max_cutoff)
        raise TypeError("cutoff should be a float or a tuple of two floats")

    @staticmethod
    def _setup_Q(Q: float | tuple) -> tuple:
        if isinstance(Q, (float, int)):
            Q = float(Q)
            return (Q, Q)
        if isinstance(Q, tuple) and len(Q) == 2:
            min_Q = float(Q[0])
            max_Q = float(Q[1])
            if min_Q > max_Q:
                raise ValueError("Q min must be <= max")
            return (min_Q, max_Q)
        raise TypeError("Q should be a float or a tuple of two floats")

    @staticmethod
    def get_params(cutoff: tuple[float, float], Q: tuple[float, float]) -> tuple[float, float]:
        """Get the cutoff frequency and Q parameters for the transform."""
        min_cutoff, max_cutoff = cutoff
        if min_cutoff == max_cutoff:
            cutoff = min_cutoff
        else:
            cutoff = float(empty(1).uniform_(min_cutoff, max_cutoff).item())
        min_Q, max_Q = Q
        if min_Q == max_Q:
            Q = min_Q
        else:
            Q = float(empty(1).uniform_(min_Q, max_Q).item())
        return cutoff, Q

    def forward(self, waveform: Tensor) -> Tensor:
        """
        Parameters
        ----------
        waveform: Tensor
            Tensor of shape [T] or [B, T].

        Returns
        -------
        out: Tensor
            Tensor with shape [B, T].
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        cutoff, Q = self.get_params(self.cutoff, self.Q)
        out = torchaudio.functional.highpass_biquad(
            waveform,
            self.sample_rate,
            cutoff_freq=cutoff,
            Q=Q,
        )
        return out

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__} (cutoff={self.cutoff}, Q={self.Q})")

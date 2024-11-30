import importlib

torch, np, random = (
    importlib.import_module("torch"),
    importlib.import_module("numpy"),
    importlib.import_module("random"),
)


def preprocess_for_WavCaps(sound_instance, device=None):
    # padding if clip is less 10 secs long

    sample_rate = 32000
    length_in_s = 10
    length = length_in_s * sample_rate

    sound_instance = torch.tensor(sound_instance).unsqueeze(0).cuda()
    if sound_instance.shape[-1] < length:
        pad_length = length - sound_instance.shape[-1]
        sound_instance = torch.nn.functional.pad(
            sound_instance, [0, pad_length], "constant", 0.0
        )

    # random cropping if clip is more than 10 secs long
    if sound_instance.shape[-1] > length:
        max_start = sound_instance.shape[-1] - length
        start = random.randint(0, max_start)
        sound_instance = sound_instance[0, start : start + length].unsqueeze(0)

    return sound_instance


def preprocess_for_AudioCLIP(sound_instance, device):
    return torch.from_numpy(sound_instance.reshape(1, -1)).to(device)


def preprocess_for_CLAP(sound_instance, device):
    def int16_to_float32(x):
        return (x / 32767.0).astype(np.float32)

    def float32_to_int16(x):
        x = np.clip(x, a_min=-1.0, a_max=1.0)
        return (x * 32767.0).astype(np.int16)

    audio_data = sound_instance.reshape(1, -1)  # Make it (1,T) or (N,T)
    audio_data = (
        torch.from_numpy(int16_to_float32(float32_to_int16(audio_data)))
        .float()
        .to(device)
    )

    return audio_data

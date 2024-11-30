def load_WavCaps(cuda_available, device, pt_file):
    from ruamel import yaml
    from WavCaps.retrieval.models.ase_model import ASE
    from torch import load

    with open(
        "/mnt/2023-audiocaptioning-msc-stefan/audio_captioning/clip/WavCaps/retrieval/settings/inference.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)

    clip = ASE(config)
    if cuda_available:
        # clip=nn.DataParallel(clip, device_ids=[0,1,2]).to(device)
        clip = clip.to(device)
    cp_path = pt_file
    cp = load(cp_path)

    # BUGFIX: This parameter does not want to load into the model.
    import torch

    assert (
        clip.text_encoder.text_encoder.embeddings.position_ids.cpu()
        == torch.arange(512)
    ).all()
    del cp["model"]["text_encoder.text_encoder.embeddings.position_ids"]

    clip.load_state_dict(cp["model"])
    clip.eval()

    return clip


def load_AudioClip(cuda_available, device, pt_file):
    from AudioCLIP.model import AudioCLIP

    clip = AudioCLIP(pretrained=pt_file)
    if cuda_available:
        clip = clip.to(device)
    clip.eval()
    # DISCLAIMER: in the demo, eval mode is not activated!

    return clip


def load_CLAP(cuda_available, device, pt_file):
    import laion_clap

    if cuda_available:
        clip = laion_clap.CLAP_Module(enable_fusion=True, device=device)

    else:
        clip = laion_clap.CLAP_Module(enable_fusion=True, device="cpu")

    # create encode_text and encode_audio function
    # IM MAGIC REPO LAUFEN LASSEN!!
    clip.load_ckpt(ckpt=pt_file, model_id=3)
    clip.eval()

    # DISCLAIMER: in the package example, eval mode is not activated!

    setattr(clip, "encode_audio", clip.get_audio_embedding_from_data)
    setattr(clip, "encode_text", clip.get_text_embedding)

    return clip

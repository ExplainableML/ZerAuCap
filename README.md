<div align="center">

# Zero-shot audio captioning with audio-language model guidance and audio context keywords

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.2305.14930-B31B1B.svg)](https://arxiv.org/abs/2305.149309) -->
[![NeurIPS - ML for Audio Workshop](http://img.shields.io/badge/NeurIPS_2023_ML_for_Audio_Workshop_(Oral)-2023-4b44ce.svg)](https://papers.nips.cc/paper/2030)
<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zero-shot-audio-captioning-with-audio/zero-shot-audio-captioning-on-clotho)](https://paperswithcode.com/sota/zero-shot-audio-captioning-on-clotho?p=zero-shot-audio-captioning-with-audio)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zero-shot-audio-captioning-with-audio/zero-shot-audio-captioning-on-audiocaps)](https://paperswithcode.com/sota/zero-shot-audio-captioning-on-audiocaps?p=zero-shot-audio-captioning-with-audio)
<br>
[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

</div>

## Description

This repository is the official implementation of the **NeurIPS 2023 - Machine Learning for Audio Workshop (Oral)** _Zero-shot audio captioning with audio-language model guidance and audio context keywords_ by Leonard Salewski, Stefan Fauth, A. Sophia Koepke, and Zeynep Akata from the University of Tübingen and the Tübingen AI Center.

## Abstract

<p align="justify">
Zero-shot audio captioning aims at automatically generating descriptive textual captions for audio content without prior training for this task. Different from speech recognition which translates audio content that contains spoken language into text, audio captioning is commonly concerned with ambient sounds, or sounds produced by a human performing an action. Inspired by zero-shot image captioning methods, we propose ZerAuCap, a novel framework for summarising such general audio signals in a text caption without requiring task-specific training. In particular, our framework exploits a pre-trained large language model (LLM) for generating the text which is guided by a pre-trained audio-language model to produce captions that describe the audio content. Additionally, we use audio context keywords that prompt the language model to generate text that is broadly relevant to sounds.
Our proposed framework achieves state-of-the-art results in zero-shot audio captioning on the AudioCaps and Clotho datasets.
</p>

## Code

Code is coming soon.

## Citation

Please cite our work with the following bibtex key.

```bib
@article{Salewski2023ZeroShotAudio,
  title   = {Zero-shot audio captioning with audio-language model guidance and audio context keywords},
  author  = {Leonard Salewski and Stefan Fauth and A. Sophia Koepke and Zeynep Akata},
}
```

## Funding and Acknowledgments

The authors thank IMPRS-IS for supporting Leonard Salewski. This work was partially funded by the BMBF Tübingen AI Center (FKZ: 01IS18039A), DFG (EXC number 2064/1 – Project number 390727645), and ERC (853489-DEXIM).

## License

This repository is licensed under the MIT License.

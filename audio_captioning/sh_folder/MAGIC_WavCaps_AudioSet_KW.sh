conda activate /mnt/miniconda3/envs/ZSAAC_7
CUDA_VISIBLE_DEVICES="1" python ../inference_magic.py\
    --language_model_code_path ../language_model/\
    --language_model_name facebook/opt-1.3b\
    --audio_code_path ../clip/\
    --audio_pt_file ../clip/WavCaps/retrieval/assets/HTSAT-BERT-PT.pt\
    --AudioCaps_inference_file_prefix ../softlinks/AudioCaps_data/\
    --clotho_inference_file_prefix ../softlinks/evaluation_data_files/\
    --GT_captions_AudioCaps ../data/AudioCaps/AudioCaps_test.json\
    --GT_captions_clotho ../data/Clotho/clotho_v2.1_test.json\
    --decoding_len 78\
    --sample_rate 32000\
    --k 45\
    --save_name MAGIC_WavCaps_AudioSet_KW\
    --include_prompt_magic False\
    --experiment test_performance\
    --path_to_AudioSet_keywords ../data/AudioSet/class_labels_indices.csv
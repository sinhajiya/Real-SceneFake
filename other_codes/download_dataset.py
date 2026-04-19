from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sinhajiya/Real_SceneFake",
    repo_type="dataset",
    local_dir="./Real_SceneFake",
    local_dir_use_symlinks=False,
    ignore_patterns=["pretrained_models/*"]
)

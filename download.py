from huggingface_hub import snapshot_download

def download_model(name, organization):
    snapshot_download(repo_id=f"{organization}/{name}", local_dir=f"models/{name}")
    print(f"Model downloaded to models/{name}")

download_model("Ouro-1.4B", "ByteDance")
download_model("Ouro-1.4B-Thinking", "ByteDance")
download_model("Ouro-2.6B", "ByteDance")
download_model("Ouro-2.6B-Thinking", "ByteDance")

download_model("Recurrent-Llama-3.2-train-recurrence-32", "smcleish")
download_model("Recurrent-TinyLlama-3T-train-recurrence-32", "smcleish")
download_model("Recurrent-OLMo-2-0425-train-recurrence-32", "smcleish")

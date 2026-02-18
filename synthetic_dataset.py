import os
import numpy as np
from datasets import load_dataset, DatasetDict
from huggingface_hub import login  # 로그인 모듈 추가

# ==========================================
# 🛑 [필수 수정] 여기에 Write 권한 토큰을 붙여넣으세요
# 예: HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
HF_TOKEN = "여기에_복사한_토큰을_붙여넣으세요"
# ==========================================

# 코드 실행 시 자동으로 로그인 수행 (터미널 설정 무시)
print(f"🔐 Logging in with token: {HF_TOKEN[:5]}...")
login(token=HF_TOKEN)

def create_and_upload_noisy_dataset(
    dataset_name: str,
    hf_username: str, 
    noise_rates: list = [0.1, 0.2, 0.3],
    seed: int = 42,
    private: bool = True 
):
    print(f"Loading dataset: {dataset_name}...")
    try:
        ds = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return

    sanitized_name = dataset_name.split('/')[-1]

    for rate in noise_rates:
        noise_percent = int(rate * 100)
        print(f"\nProcessing noise rate: {noise_percent}%")
        
        noisy_ds_dict = DatasetDict()
        
        for split in ds.keys():
            if split != 'train':
                noisy_ds_dict[split] = ds[split]
                continue
            
            original_data = ds[split]
            n_samples = len(original_data)
            
            np.random.seed(seed)
            flip_mask = np.random.rand(n_samples) < rate
            
            def inject_noise(example, idx):
                if flip_mask[idx]:
                    return {
                        "chosen": example["rejected"],
                        "rejected": example["chosen"],
                    }
                else:
                    return {
                        "chosen": example["chosen"],
                        "rejected": example["rejected"],
                    }

            noisy_split = original_data.map(
                inject_noise,
                with_indices=True,
                desc=f"Injecting {rate} noise into {split}"
            )
            
            noisy_ds_dict[split] = noisy_split

        # 저장소 이름 생성
        target_repo_id = f"{hf_username}/{sanitized_name}-noise-{noise_percent}"
        
        print(f"  ☁️ Uploading to {target_repo_id}...")
        try:
            noisy_ds_dict.push_to_hub(
                target_repo_id,
                private=private
            )
            print(f"  ✅ Successfully uploaded: https://huggingface.co/{target_repo_id}")
            
        except Exception as e:
            print(f"  ❌ Failed to upload {target_repo_id}: {e}")
            print("  👉 팁: 토큰이 'Write' 권한인지, 'promotion' 조직에 속해 있는지 확인하세요.")

# --- 실행 ---

# [중요] promotion이 본인 아이디인지 조직(Organization)인지 확인 필요
# 조직이라면 본인 계정이 그 조직의 Member여야 함.
MY_HF_USERNAME = "promotion"  

# 1. HH-RLHF
create_and_upload_noisy_dataset("Anthropic/hh-rlhf", hf_username=MY_HF_USERNAME)

# 2. UltraFeedback
create_and_upload_noisy_dataset("HuggingFaceH4/ultrafeedback_binarized", hf_username=MY_HF_USERNAME)
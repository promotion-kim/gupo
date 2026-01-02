import os
import glob
import re
import argparse  # 인자 입력을 위해 추가
from huggingface_hub import HfApi

def upload_final_safe(base_output_dir, hf_id, token=None):
    api = HfApi(token=token)
    
    # 1. outputs 폴더 스캔
    all_dirs = glob.glob(os.path.join(base_output_dir, "*"))
    
    # 타임스탬프 패턴 인식
    timestamp_pattern = re.compile(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d+)')

    print(f"🔍 Scanning '{base_output_dir}'...")

    for folder_path in all_dirs:
        if not os.path.isdir(folder_path):
            continue

        dirname = os.path.basename(folder_path)
        
        # [안전장치 1] 타임스탬프 없으면 스킵
        match = timestamp_pattern.search(dirname)
        if not match:
            continue
            
        clean_exp_name = dirname[:match.start()]

        # [안전장치 2] LATEST 폴더 없으면 스킵
        latest_dir = os.path.join(folder_path, "LATEST")
        if not os.path.exists(latest_dir):
            continue

        # --- 저장소 생성 ---
        repo_name = f"{hf_id}/{clean_exp_name}"
        try:
            # ⭐️ [수정됨] private=False로 설정하여 Public Repo로 생성
            api.create_repo(repo_id=repo_name, private=False, exist_ok=True)
            print(f"\n🚀 Processing: {clean_exp_name}")
            print(f"   Target: https://huggingface.co/{repo_name}")
        except Exception as e:
            print(f"❌ Repo Error: {e}")
            continue

        # --- 파일 업로드 ---
        
        # 1. Policy
        policy_path = os.path.join(latest_dir, "policy.pt")
        if os.path.exists(policy_path):
            target_name = f"{clean_exp_name}_llm.pt"
            print(f"   📤 Uploading policy.pt -> {target_name}")
            api.upload_file(
                path_or_fileobj=policy_path,
                path_in_repo=target_name,
                repo_id=repo_name,
                commit_message=f"Upload Policy"
            )

        # 2. MLP
        mlp_path = os.path.join(latest_dir, "mlp.pt")
        if os.path.exists(mlp_path):
            target_name = f"{clean_exp_name}_mlp.pt"
            print(f"   📤 Uploading mlp.pt    -> {target_name}")
            api.upload_file(
                path_or_fileobj=mlp_path,
                path_in_repo=target_name,
                repo_id=repo_name,
                commit_message=f"Upload MLP"
            )

        # 3. Config
        config_yaml = os.path.join(folder_path, "config.yaml")
        config_json = os.path.join(folder_path, "config.json")
        
        if os.path.exists(config_yaml):
            print(f"   📤 Uploading config.yaml")
            api.upload_file(
                path_or_fileobj=config_yaml,
                path_in_repo="config.yaml",
                repo_id=repo_name,
                commit_message="Upload config.yaml"
            )
        elif os.path.exists(config_json):
            print(f"   📤 Uploading config.json")
            api.upload_file(
                path_or_fileobj=config_json,
                path_in_repo="config.json",
                repo_id=repo_name,
                commit_message="Upload config.json"
            )

    print("\n✅ Upload finished!")

if __name__ == "__main__":
    # ⭐️ [수정됨] 터미널에서 인자를 받도록 설정
    parser = argparse.ArgumentParser(description="Upload models to Hugging Face Public Repo")
    
    # 필수 인자: 토큰, 경로, 아이디
    parser.add_argument("--token", type=str, required=True, help="Hugging Face Write Token")
    parser.add_argument("--dir", type=str, required=True, help="Base output directory path")
    parser.add_argument("--id", type=str, default="promotion", help="Hugging Face User ID (default: promotion)")

    args = parser.parse_args()

    # 입력받은 인자로 함수 실행
    upload_final_safe(args.dir, args.id, token=args.token)
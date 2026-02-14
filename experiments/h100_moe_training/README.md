# 🚀 H100 MoE Training on Vessl AI

이 문서는 Vessl AI의 H100 인스턴스를 사용하여 야옹이 모델(Yaongi-Nemotron-Teon)을 학습시키는 통합 가이드입니다.

## 1. Vessl AI 환경 설정

### 🖥️ 하드웨어 사양 및 비용
- **GPU**: NVIDIA H100 SXM (80GB VRAM)
- **비용**: 약 **$2.39/hr** (크레딧 사용 권장)
- **권장 사양**: 
  - CPU: 16 vCPU 이상
  - RAM: 60GB 이상
  - Disk: 100GB 이상 (데이터셋 캐싱 및 체크포인트 저장용)

### 🐳 Docker 이미지 및 Mamba 지원
H100(Hopper) 아키텍처 지원 및 Mamba-SSM 최적화를 위해 아래 이미지를 사용합니다.
- **Base Image**: `nvcr.io/nvidia/pytorch:23.10-py3` (CUDA 12.2+, PyTorch 2.1+)
- **SSM 최적화**: Dockerfile 내에서 `mamba-ssm`을 설치하여 GPU 가속을 지원합니다.
- **Vessl 설정**: 워크스페이스 생성 시 Custom Image로 위 도커파일을 빌드하거나, 유사한 `pytorch:2.3.1-cuda12.1` 등을 선택 후 스크립트가 필요한 패키지를 추가 설치하게 됩니다.

## 2. 필수 계정 및 키 설정 (중요!)

학습 로그 기록(WandB)과 모델 업로드(HuggingFace)를 위해 API 키 설정이 필요합니다.

### 2-1. WandB (Weights & Biases) 계정 설정
1. [WandB 사이트](https://wandb.ai/site)에 접속하여 회원가입을 합니다.
2. 로그인 후 우측 상단 프로필 -> **User Settings** -> **API keys** 로 이동합니다.
3. 키를 복사해둡니다. (예: `d4e5f...`)

### 2-2. HuggingFace 토큰 설정
1. [HuggingFace Settings](https://huggingface.co/settings/tokens)에서 **Access Token**을 생성합니다.
2. `Write` 권한이 있는 토큰을 생성하고 복사합니다.

### 2-3. `.env` 파일 생성
프로젝트 루트(`Open-Yaongi/`)에 `.env` 파일을 만들고 아래와 같이 키를 붙여넣으세요. 이 파일은 git에 업로드되지 않도록 주의하세요 (이미 .gitignore에 포함됨).

```bash
# .env 파일 예시
WANDB_API_KEY=d4e5f6g7h8...
HF_TOKEN=hf_AbCdEf...
```

## 3. 실행 방법 (원클릭)

Vessl 워크스페이스 터미널에서 다음 명령어를 실행하면 **환경설정부터 학습까지 한 번에** 진행됩니다.

```bash
# 1. 레포지토리 클론 (이미 되어있다면 생략)
git clone https://github.com/gyunggyung/Open-Yaongi.git
cd Open-Yaongi

# 2. 실행 스크립트 권한 부여
chmod +x run_all_vessl.sh

# 3. 통합 실행 (WandB 키 설정 권장)
export WANDB_API_KEY="your_api_key_here"
./run_all_vessl.sh
```

### `run_all_vessl.sh` 내부 동작 순서
1.  **환경 설정**: `pip install`로 필요한 라이브러리(`transformers`, `datasets` 등) 설치
2.  **데이터 다운로드**: `download_datasets.py` 실행 -> `./data_cache` 폴더 생성
3.  **토크나이저 학습**: `train_tokenizer.py` 실행 -> BPE 토크나이저 생성 (`./custom_tokenizer`)
4.  **모델 학습**: `train_h100_moe.py` 실행 -> H100에서 고속 학습 시작

## 3. 학습 모니터링
- **WandB**: 스크립트 시작 시 API Key를 입력했다면 WandB 대시보드에서 실시간 Loss 확인 가능
- **리소스 사용량**: Vessl 대시보드 또는 터미널에서 `nvidia-smi`로 GPU 로드율 확인 (H100 점유율 확인 필수)

## 4. 결과물 저장
학습이 완료되면 다음 경로에 결과물이 저장됩니다:
- **체크포인트**: `./checkpoints/checkpoint_{step}`
- **토크나이저**: `./custom_tokenizer`

> **주의**: Vessl AI의 인스턴스가 종료되면 데이터가 사라질 수 있으므로, **Persistent Storage**(`/root/files` 등)에 결과물을 저장하거나 HuggingFace Hub 업로드 기능을 활성화하세요 (`train_h100_moe.py` 내 `push_to_hub` 옵션).

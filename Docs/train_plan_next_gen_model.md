# 🚀 H100 (1-GPU) Nemotron-Style Hybrid MoE 학습 상세 계획

> **기반 논문**: NVIDIA Nemotron-3 Nano (ArXiv 2512.20848) & Kakao Tech (Kanana)
> **목표**: H100 80GB 1대에서 **4.2B (Active 0.6B)** 모델을 FP8 + Muon으로 초고속 학습.

---

## 1. 🧠 모델 아키텍처 상세 (Layer-by-Layer)

**총 52 Layer**, **Nemotron-3 Nano**의 **Interleaved** 구조를 따르며, **Squared ReLU**와 **No Bias**를 적용하여 성능을 극대화합니다.

| Layer Index | Type | Detail | 비고 |
|:---:|:---:|:---|:---|\
| **Embed** | Embedding | Vocab: 32,768 (Special Tokens 포함) | Output: 2048 dim |\
| **0** | **Mamba-2 + MoE** | Experts: 64, Top-K: 6, **SqReLU**, **No Bias** | Low-level Features |\
| **1** | **Mamba-2 + MoE** | Experts: 64, Top-K: 6, **SqReLU**, **No Bias** | |\
| **2** | **Engram (Short)** | N-gram: 3, Vocab: 200k | **단기 기억 (Context)** |\
| **3 ~ 4** | **Mamba-2 + MoE** | Experts: 64, Top-K: 6, **SqReLU**, **No Bias** | |\
| **5** | **Attention (GQA)** | Heads: 32, KV: 8, **SqReLU**, **No Bias** | **Long-range Dependency** |\
| **6 ~ 10** | **Mamba-2 + MoE** | Experts: 64, Top-K: 6, **SqReLU**, **No Bias** | |\
| **11** | **Attention (GQA)** | Heads: 32, KV: 8, **SqReLU**, **No Bias** | |\
| **12 ~ 16** | **Mamba-2 + MoE** | Experts: 64, Top-K: 6 | |\
| **17** | **Attention (GQA)** | Heads: 32, KV: 8 | |\
| ... | ... | (반복 패턴) | |\
| **26** | **Engram (Long)** | N-gram: 4, Vocab: 500k | **장기 지식 (Knowledge)** |\
| ... | ... | (반복 패턴) | |\
| **51** | **Mamba-2 + MoE** | Experts: 64, Top-K: 6, **SqReLU**, **No Bias** | Final Abstract Features |\
| **Head** | Linear | 2048 -> 32,768, **No Bias** | Final Prediction |\

> **핵심 변경 사항 (Nemotron Recipe)**:
> 1.  **Squared ReLU**: 일반 ReLU 대신 `relu(x)^2` 사용. MoE 학습 시 전문가(Expert)들의 활성화를 돕고 수렴을 가속화합니다.
> 2.  **No Bias**: 모든 Linear, LayerNorm 레이어에서 편향(Bias) 제거. 메모리 사용량 감소 및 연산 속도 증가.
> 3.  **GQA Interleaving**: 6개 레이어마다 1개의 Attention 레이어 배치 (Mamba의 약점인 Recall 보완).

---

## 2. ⚡ 최적화 전략 (Muon + FP8 + WSD)

**Kakao**와 **NVIDIA**가 증명한 최신 기법을 모두 적용합니다.

### A. Optimizer: Muon (Momentum Orthogonalized)
*   **적용 대상**: 모든 2D Tensor (Linear Weights, Experts).
*   **제외 대상**: 1D Tensor (Embeddings, RMSNorm weights) -> **AdamW** 사용.
*   **효과**: AdamW 대비 **2배 빠른 수렴 속도** 및 **VRAM 절약**.

### B. Precision: FP8 Hybrid
*   **라이브러리**: `transformer_engine` (NVIDIA 공식)
*   **전략**:
    *   **Weights/Activations**: **E4M3** (정밀도 중심)
    *   **Gradients**: **E5M2** (범위 중심)
    *   **Autocast**: `te.fp8_autocast`로 자동 처리.

### C. Scheduler: WSD (Warmup-Stable-Decay)
*   **Warmup**: 초기 1% 구간 급격 상향.
*   **Stable**: 전체의 **80%** 구간 동안 **Max Learning Rate (2e-3)** 유지. (Muon 특성상 높은 LR 가능)
*   **Decay**: 마지막 **20%** 구간에서 급격히 **1e-5**까지 감쇠. (수렴 품질 결정)

---

## 3. 💾 데이터셋 믹스 (비율 확정)

**Total 15B Tokens** (초기 실험용). 추후 25T까지 확장 가능.

| 종류 | 비율 | 데이터셋 소스 (경로) | 목적 |
|---|---|---|---|
| **한국어** | **40%** | `HAERAE-HUB/KOREAN-SyntheticText-1.5B` 등 | 고품질 한국어 능력 |
| **영어** | **20%** | `allenai/dolma3` (OLMo-3), `FineWeb-Edu` | 범용 지식 및 논리 |
| **Code** | **25%** | `Stack-Edu`, `Dolci-Think-SFT` (CoT) | 추론 능력 및 Coding |
| **Math** | **15%** | `HuggingFaceTB/finemath` | 수리 논리력 |

---

## 4. 🛠️ 실행 가이드 (H100)

이 코드는 **H100 (Sm_90)** 아키텍처에 최적화되어 있습니다. 기존 A100/T4에서는 `FP8` 기능이 자동으로 꺼지고 `BF16`으로 동작합니다.

```bash
# 1. 도커 이미지 빌드 (필수)
docker build -t yaongi-h100 .

# 2. 컨테이너 실행 (Shm 사이즈 중요)
docker run --gpus all -it --ipc=host --ulimit stack=67108864 -v $(pwd):/workspace yaongi-h100

# 3. 데이터 다운로드
python download_datasets.py

# 4. 학습 시작 (FP8 & Muon 자동 적용)
python train_h100_moe.py
```

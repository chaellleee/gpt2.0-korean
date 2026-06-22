# 한국어 문학으로 학습한 GPT

Andrej Karpathy의 강의 **["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY)** (nn-zero-to-hero, Lecture 7)의
디코더 전용 Transformer를 처음부터 구현하고, **tiny-Shakespeare 대신 공개 한국어 근대문학**으로 학습시킨 프로젝트입니다.
모델 구조는 강의와 동일하며, 바뀐 것은 데이터(영어 → 한국어)입니다.

## 저장소 구조

```
gpt2-korean/
├── gpt.py              # 학습 스크립트 (data/input.txt 를 읽음)
├── quicktest.py        # 1분 자가검증
├── requirements.txt
├── data/
│   └── input.txt       # 한국어 학습 말뭉치 (위키문헌 공개 작품 39편, 약 52만 자)
├── notebooks/
│   ├── GPT_korean_ONECELL.ipynb   # ★ Colab 추천 — 셀 1개로 전부 실행
│   ├── GPT_korean_colab.ipynb     # char-level 단계별 버전
│   └── GPT_korean_BPE_colab.ipynb # 직접 구현한 BPE 토크나이저 버전 (보너스)
├── results/
│   └── loss_curve.png  # 학습 손실 곡선
└── docs/
    └── DEVLOG.md       # 개발 과정 · 과적합 등 시행착오 기록
```

## 실행

**A. Colab (권장)** — `notebooks/GPT_korean_ONECELL.ipynb` 를 업로드 → `런타임 유형 = GPU(T4)` → 셀 실행.
데이터가 노트북에 내장되어 별도 업로드가 필요 없습니다.

**B. 로컬 (venv)**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python quicktest.py     # 1분 자가검증 (학습이 되는지 PASS/FAIL)
python gpt.py           # 정식 학습 (GPU 권장)
```

## 데이터셋

`data/input.txt` 는 [한국어 위키문헌](https://ko.wikisource.org)의 **퍼블릭 도메인** 한국 근대문학
(김유정·현진건·김동인·이상·나도향·이효석·최서해·강경애 등 39편)을 마크업 제거 후 이어 붙인 문자 단위 말뭉치입니다.

## 결과

손실이 랜덤 수준(`ln(vocab)≈7.9`)에서 시작해 감소하며, 학습된 모델은 한국 근대문학 문체의 텍스트를 생성합니다.
자세한 곡선·샘플과 개발 과정은 `results/`, `docs/DEVLOG.md` 참고.

## 참고

- A. Karpathy, *Let's build GPT* — [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)
- Vaswani et al., *Attention Is All You Need* (2017)
- 데이터 출처: 한국어 위키문헌 (퍼블릭 도메인)

코드 MIT License · 데이터 퍼블릭 도메인

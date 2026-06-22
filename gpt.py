"""
GPT — 디코더 전용 Transformer를 처음부터 구현 (Karpathy "Let's build GPT", Lecture 7).

강의와 동일한 구조(토큰+위치 임베딩, causal mask 멀티헤드 셀프어텐션, 잔차 연결,
LayerNorm, 피드포워드 블록)이며, tiny-Shakespeare 대신 공개 한국어 근대문학
(김유정·현진건·김동인·이상·나도향·이효석 등, 위키문헌)을 문자 단위로 학습한다.

실행:
    python gpt.py                                  # 기본 설정으로 학습 후 샘플 생성
    python gpt.py --max_iters 1000                 # 빠른 학습
    python gpt.py --sample_only --resume ckpt.pt   # 학습된 모델로 생성만

산출물:
    ckpt.pt          모델 가중치 + 설정 + 어휘
    loss_curve.png   학습/검증 손실 곡선
    samples.txt      모델이 생성한 텍스트

GPU(CUDA) / Apple MPS / CPU를 자동 감지한다.
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.nn import functional as F


# ----------------------------------------------------------------------------
# 설정 (명령줄 인자로 덮어쓸 수 있음)
# ----------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="한국어 문학으로 GPT 학습 (Karpathy Lecture 7)")
    p.add_argument("--data",       type=str,   default="data/input.txt", help="학습 텍스트 경로")
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--block_size", type=int,   default=256, help="문맥 길이(한 번에 보는 토큰 수)")
    p.add_argument("--n_embd",     type=int,   default=384, help="임베딩/모델 차원")
    p.add_argument("--n_head",     type=int,   default=6,   help="어텐션 헤드 수")
    p.add_argument("--n_layer",    type=int,   default=6,   help="Transformer 블록 수")
    p.add_argument("--dropout",    type=float, default=0.2)
    p.add_argument("--max_iters",  type=int,   default=5000)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int,   default=200)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--seed",       type=int,   default=1337)
    p.add_argument("--out",        type=str,   default="ckpt.pt")
    p.add_argument("--max_new_tokens", type=int, default=2000, help="학습 후 생성할 토큰 수")
    p.add_argument("--resume",     type=str,   default=None, help="이어서/생성에 쓸 체크포인트")
    p.add_argument("--sample_only", action="store_true", help="학습 건너뛰고 생성만")
    p.add_argument("--device",     type=str,   default=None, help="cuda|mps|cpu (미지정 시 자동)")
    return p.parse_args()


def pick_device(requested=None):
    """사용할 장치를 자동 선택한다 (GPU > MPS > CPU)."""
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------------------------------------------------------
# 모델 — 강의와 동일한 구조
# ----------------------------------------------------------------------------
class Head(nn.Module):
    """causal 셀프어텐션 헤드 1개."""
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 미래 토큰을 못 보게 가리는 하삼각 마스크
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)            # (B,T,hs)
        q = self.query(x)          # (B,T,hs)
        # 어텐션 점수 = Q·Kᵀ, √dₖ로 스케일
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # 미래 차단
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)          # (B,T,hs)
        return wei @ v             # (B,T,hs)


class MultiHeadAttention(nn.Module):
    """여러 개의 헤드를 병렬로 돌리고 결과를 합친다."""
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """위치별 MLP (4배로 키웠다가 다시 줄임)."""
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer 블록: 어텐션(소통) + MLP(연산)."""
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # 잔차 연결 + pre-norm 어텐션
        x = x + self.ffwd(self.ln2(x))   # 잔차 연결 + pre-norm MLP
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)      # 토큰 임베딩
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # 위치 임베딩
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)               # 마지막 LayerNorm
        self.lm_head = nn.Linear(n_embd, vocab_size)   # 다음 토큰 확률(logits) 출력
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """가중치 초기화 (std=0.02). 초기 손실이 ln(vocab)에서 시작하도록."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        tok_emb = self.token_embedding_table(idx)                                # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb            # 토큰 + 위치 정보를 더함
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)         # (B,T,vocab)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """첫 토큰부터 시작해 한 글자씩 자기회귀적으로 생성."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]      # 문맥을 block_size로 자름
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                 # 마지막 위치의 예측만 사용
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # 확률에 따라 1개 샘플
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ----------------------------------------------------------------------------
# 학습
# ----------------------------------------------------------------------------
def main():
    args = get_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    print(f"device = {device}")

    # ---- 데이터 + 토크나이저 (강의와 동일한 문자 단위) ----
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}      # 문자 -> 정수
    itos = {i: ch for i, ch in enumerate(chars)}      # 정수 -> 문자
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join(itos[i] for i in l)
    print(f"데이터: {len(text):,}자 | 어휘 크기: {vocab_size}")

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))                          # 9:1 로 학습/검증 분할
    train_data, val_data = data[:n], data[n:]

    def get_batch(split):
        """무작위 위치에서 (입력 x, 정답 y=한 칸 뒤) 배치를 뽑는다."""
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - args.block_size, (args.batch_size,))
        x = torch.stack([d[i:i + args.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + args.block_size] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model):
        """학습/검증 손실을 여러 번 평균내어 측정 (과적합 확인용)."""
        out = {}
        model.eval()
        for split in ("train", "val"):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # ---- 모델 생성 / 이어받기 ----
    model = GPT(vocab_size, args.n_embd, args.n_head, args.n_layer, args.block_size, args.dropout).to(device)
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        itos = ck.get("itos", itos)
        decode = lambda l: "".join(itos[i] for i in l)
        print(f"체크포인트 이어받음: {args.resume}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터: {n_params/1e6:.2f} M")

    history = {"iter": [], "train": [], "val": []}

    if not args.sample_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for it in range(args.max_iters + 1):
            # 주기적으로 손실 측정 + 기록
            if it % args.eval_interval == 0 or it == args.max_iters:
                losses = estimate_loss(model)
                history["iter"].append(it)
                history["train"].append(losses["train"])
                history["val"].append(losses["val"])
                print(f"step {it:5d}: train {losses['train']:.4f}, val {losses['val']:.4f}")
            # 한 스텝 학습: 배치 -> 손실 -> 역전파 -> 업데이트
            xb, yb = get_batch("train")
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        torch.save({"model": model.state_dict(),
                    "config": vars(args),
                    "stoi": stoi, "itos": itos,
                    "history": history}, args.out)
        print(f"체크포인트 저장 -> {args.out}")
        save_loss_curve(history, "loss_curve.png")

    # ---- 생성 ----
    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # 빈 문맥에서 시작
    out = decode(model.generate(context, max_new_tokens=args.max_new_tokens)[0].tolist())
    with open("samples.txt", "w", encoding="utf-8") as f:
        f.write(out)
    print("\n===== 생성 샘플 =====\n")
    print(out[:1500])
    print("\n전체 샘플은 samples.txt에 저장됨")


def save_loss_curve(history, path):
    """학습/검증 손실 곡선을 그려 저장한다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4.5))
        plt.plot(history["iter"], history["train"], label="train", marker="o", ms=3)
        plt.plot(history["iter"], history["val"], label="val", marker="o", ms=3)
        plt.xlabel("iteration"); plt.ylabel("cross-entropy loss")
        plt.title("GPT on Korean literature — training curve")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(path, dpi=120)
        print(f"손실 곡선 저장 -> {path}")
    except Exception as e:
        print(f"(손실 곡선 저장 실패: {e})")


if __name__ == "__main__":
    main()

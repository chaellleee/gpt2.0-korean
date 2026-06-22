"""
quicktest.py — fast sanity check that GPT 2.0 is wired up correctly.

It trains a TINY model for a few hundred steps on input.txt (runs on a CPU in
~1 minute) and checks three things:

  1. initial loss is close to ln(vocab_size)  (i.e. the model starts at chance)
  2. loss goes DOWN by a clear margin after training
  3. the generated sample actually contains Hangul characters

Prints PASS / FAIL for each. This does NOT produce a good model — it only proves
the pipeline learns. For real results use:  python gpt.py
"""
import math, torch
ns = {}
exec(open("gpt.py").read().split("def main()")[0], ns)   # import the model classes
GPT = ns["GPT"]

torch.manual_seed(1337)
text = open("data/input.txt", encoding="utf-8").read()
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data)); train, val = data[:n], data[n:]
blk, BS = 64, 16

def batch(split):
    d = train if split == "train" else val
    ix = torch.randint(len(d) - blk, (BS,))
    return (torch.stack([d[i:i+blk] for i in ix]),
            torch.stack([d[i+1:i+1+blk] for i in ix]))

model = GPT(V, n_embd=128, n_head=4, n_layer=3, block_size=blk, dropout=0.2)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

@torch.no_grad()
def mean_loss(split, k=20):
    model.eval(); s = 0.0
    for _ in range(k):
        x, y = batch(split); _, l = model(x, y); s += l.item()
    model.train(); return s / k

print(f"vocab size = {V}   random-guess loss = ln(V) = {math.log(V):.3f}")
loss0 = mean_loss("val")
print(f"step   0: val loss {loss0:.3f}")
STEPS = 300
for it in range(1, STEPS + 1):
    x, y = batch("train"); _, loss = model(x, y)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if it % 100 == 0:
        print(f"step {it:3d}: val loss {mean_loss('val'):.3f}")
loss1 = mean_loss("val")

# generate a short sample
model.eval()
ctx = torch.zeros((1, 1), dtype=torch.long)
with torch.no_grad():
    for _ in range(300):
        lg, _ = model(ctx[:, -blk:]); p = torch.softmax(lg[:, -1, :], -1)
        ctx = torch.cat((ctx, torch.multinomial(p, 1)), 1)
sample = "".join(itos[i] for i in ctx[0].tolist())
hangul = sum(1 for ch in sample if "가" <= ch <= "힣")

print("\n----- sample -----")
print(sample[:200])
print("------------------\n")

ok1 = abs(loss0 - math.log(V)) < 0.5
ok2 = loss1 < loss0 - 1.0
ok3 = hangul > 30
print(f"[{'PASS' if ok1 else 'FAIL'}] starts at chance     (|{loss0:.2f} - {math.log(V):.2f}| < 0.5)")
print(f"[{'PASS' if ok2 else 'FAIL'}] loss decreased        ({loss0:.2f} -> {loss1:.2f})")
print(f"[{'PASS' if ok3 else 'FAIL'}] generates Hangul      ({hangul}/300 chars)")
print("\nRESULT:", "ALL PASS — the GPT trains correctly." if (ok1 and ok2 and ok3) else "check failures above")

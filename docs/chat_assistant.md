# 💬 The chat assistant, in detail

The [README](../README.md#-ask-your-roll-questions) covers how to turn it on. This is everything else:
what it costs to run, what it can and cannot do, and how to check it yourself.

## 🧰 This is a concept, and its tool belt is short

It can count, filter and group your roll, give you means, medians and quartiles, draw a histogram, explain
the statistics, and run a **ratio study**: median ratio, weighted mean ratio, COD and PRD, by city,
neighbourhood or sale year.

**Please check its figures on anything more complicated.** A question outside those tools depends on SQL a
language model wrote for itself, and that is where it goes wrong. As tools are added, this warning gets
shorter.

### What it cannot do yet

- **VEI, the Vertical Equity Indicator.** It comes to the app once the IAAO's new standards are released.
  It replaces PRB, which the IAAO is phasing out, so PRB will not be added. The notebook reports both.
- **A ratio study for a property type you did not build a model for.** The sales table only holds the types
  you modelled.
- **Scoring the model on the sales it trained on.** That is impossible on purpose, because it would flatter
  the model. The Results tab shows those figures in its Training column.
- **Following up on an answer, on Local.** It remembers your earlier questions but not its own answers, so
  *"explain that more simply"* will not work. Ask the question in full. Hosted does not have this limit. We aim
  to address this in a future version.
- **Legal, policy or appeal advice.** These are benchmark estimates.

## 📐 The ratio study matches the Results tab

To four decimals, on all four property types. It scores the model only on sales the model never saw, and it
scores the model and the county on the *same* sales, because anything else flatters one of them.

That matters more than it sounds. Scoring the model on every sale, including the ones it trained on, makes its
COD look better than it really is.

## ⚙️ What counts as acceptable is yours to decide

The rules differ between jurisdictions and they change, so the assistant does not quote a range from memory or
from any standards body. It reads the ranges you entered on the **Settings** tab, and reports whether each
figure is inside them.

## 🔍 A small group is kept out of any ranking

A statistic from a handful of sales mostly measures the handful. Groups below the line you set on the Settings
tab (30 sales to begin with) come with a warning and are listed apart, so a city with five sales cannot come
out as the worst in the county. Below a second line (5 sales to begin with), a group gets no statistics at all.

## 🏠 The local model

Local runs [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) on your own computer. Once it is
installed, it needs no API key, no account and no internet connection. From pressing **Start Chat** to the
answer, the app tries to reach nothing at all, which was measured with every outbound connection refused.

ℹ️ One thing a county firewall will still notice: the **Build Model** tab fetches a decorative loading animation
from `lottiefiles.com`. No data goes with it, and if it is blocked the animation is simply skipped.

### Installing it

```bash
uv sync --extra slm      # PyTorch with CUDA 13 and transformers, about 3 GB
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B')"     # about 9.3 GB
```

⚠️ **A plain `uv sync` afterwards removes the extra again, silently.** Re-run it with `--extra slm`. It is quick,
because the files are cached.

The second command may warn that you are *"sending unauthenticated requests"*. That's fine. The model is public,
so you don't need an account.

To try a different model, set `LOCAL_MODEL` in your `.env`. See [`.env.example`](../.env.example).

### What it costs to run

Measured on an RTX 3080, with the Lee County roll loaded:

| | memory | speed |
|---|---|---|
| **4-bit, the default** | **3.07 GiB video memory** | about 15 tokens a second |
| bf16 | 8.57 GiB video memory | about 19 tokens a second |
| no graphics card, on the processor | 16.46 GiB **memory** | about 2.6 tokens a second |
| no graphics card, memory tight | 8.60 GiB **memory** | about 0.6 tokens a second |

It is **quantised by default, not as a fallback.** bf16 needs 8.57 GiB, which does not fit the 8 GB card this
project designs for, and costs only about 20% of the speed.

With no graphics card it still works, and the app picks whichever row fits your free memory. ⚠️ Those speeds are
from a 24-core desktop, so expect a laptop to be several times slower. A one-line answer took about two minutes
there, even on the faster row. If neither fits, the app says so rather than grinding your machine to a halt.

> ℹ️ On Windows you will see `triton not found; flop counting will not work` every time PyTorch loads. It is
> harmless. Triton has no Windows build and nothing here counts FLOPs.

## ☁️ The hosted model

Hosted uses Anthropic's Claude. It starts on **Haiku**, which is fast, inexpensive, and handles most questions an
assessor asks of a roll. The dropdown on the tab has more capable models for a question worth more thought.

Your questions, and the rows the tools return, go to Anthropic. The app never prints your key. See
[`.env.example`](../.env.example) for the settings.

## ✅ Check it yourself

The questions listed under **What can I ask it?** are not suggestions somebody thought sounded good. This script
reads that list from the app, puts each question to the assistant through its own chat box, and checks every
number in the answer against the same figure worked out separately.

```bash
uv run python scripts/check_chat_assistant.py                                  # needs no model and no key
uv run python scripts/check_chat_assistant.py --engine local                   # asks the local model
uv run python scripts/check_chat_assistant.py --engine local --keep-history    # in one long conversation
uv run python scripts/check_chat_assistant.py --engine hosted                  # the real API, a little money
```

Please read the answers, not just the score. It is evidence from one roll, not a guarantee on yours, and language
models are not perfectly repeatable: Claude Haiku has occasionally called a PRD inside the range "outside" it, even
with the tool's result saying inside. That is why the screen says to check the figures.

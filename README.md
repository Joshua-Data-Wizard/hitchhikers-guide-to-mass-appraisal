# 🐬 A Hitchhiker's Guide to Mass Appraisal

**An in-depth, honest walkthrough of building a gradient boosting model for
property valuation — on a real Florida county tax roll.**

Written for appraisers, not engineers. If you have ever wondered whether
machine learning is something your office could actually use, this repository
is here to answer that, with working code you can read.

> ### 🎯 The goal is a benchmark, not a better roll
>
> This is **not** an attempt to produce the best values ever made. It is
> something to **benchmark your own values against**, and a safe place to try
> machine learning on data you recognise.
>
> ⚠️ **The values this produces are benchmark estimates. They are not
> assessments, not defensible values, and not evidence in an appeal.**

---

## 🚀 Quickstart

You do **not** need Python installed. [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
handles the interpreter, the environment and every package.

```bash
git clone https://github.com/Joshua-Data-Wizard/hitchhikers-guide-to-mass-appraisal.git
cd hitchhikers-guide-to-mass-appraisal

uv sync                                       # 1. build the environment (~45s)
uv run python scripts/extract_data.py         # 2. unpack the tax roll (~1 min)
uv run jupyter lab                            # 3. open notebooks/ and go
```

That's it. Three commands.

### 🪟 One thing to do first, if you're on Windows

```bash
git config --global core.longpaths true
```

Windows enforces a 260-character path limit unless told otherwise, and a Python
environment nests deeply — the longest path this project creates is **281
characters**. It generally works anyway, because `uv` sidesteps the limit. But
if your Documents folder sits under a long OneDrive organisation name (say
`OneDrive - Lee County Property Appraiser`), that one command removes a whole
class of baffling "file not found" and `$GIT_DIR too big` errors before they
happen.

**Also: clone somewhere reasonably shallow.** `C:\Users\you\GitHub\` is ideal.
A folder six levels deep inside another project is asking for trouble.

### Why step 2?

The tax roll is about **1.7 GB**, and GitHub refuses any file over 100 MB — the
parcel attribute table alone is 1.08 GB. So the data ships **gzipped** (~199 MB)
and `extract_data.py` puts it back, byte for byte, verified against SHA-256
checksums. **Nothing is downloaded**; it is all already in your clone.

### Optional — run a small language model locally

**You do not need this for the walkthrough.** The notebook is entirely about
valuation and uses no language models at all.

This extra is only for one version of the app's chatbot, which can run a small
language model on your own machine with **no data leaving the building** — often
the only version an assessor's office is permitted to use:

```bash
uv sync --extra slm      # adds PyTorch w/ CUDA 13 (~3 GB) + transformers
```

⚠️ **A plain `uv sync` afterwards prunes that extra straight back out, silently.**
Re-run it with `--extra slm`; it is quick, because the wheels are cached.

Then fetch the model **once** — roughly **9.3 GB**, and `uv sync` does not do it
for you:

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B')"
```

Pick a different model by setting `LOCAL_MODEL` in your `.env` — see
`.env.example`.

In the app, the chat section has a **Hosted / Local** choice. Once the model is
fetched, Local needs no API key, no account and no internet connection, and no
parcel data leaves your computer. Measured with every outbound connection refused:
from pressing Start Chat to the answer, the app tries to reach nothing at all.

ℹ️ One thing a county firewall will still notice: the **Build Model** tab fetches a
decorative loading animation from `lottiefiles.com`. No data goes with it, and if
it is blocked the animation is simply skipped.

#### What it costs to run, measured

On an RTX 3080 with the Lee County roll loaded:

| | memory | speed |
|---|---|---|
| **4-bit — the default** | **3.07 GiB VRAM** | ~15 tokens/sec |
| bf16 | 8.57 GiB VRAM | ~19 tokens/sec |
| no GPU, on the processor | 16.46 GiB **RAM** | ~2.6 tokens/sec |
| no GPU, memory tight | 8.60 GiB **RAM** | ~0.6 tokens/sec |

It is **quantised by default, not as a fallback.** bf16 needs 8.57 GiB, which
does not fit the 8 GB card this project designs for, and costs only about 20% of
the speed.

With no GPU it still works, and the app picks whichever row fits your free memory.
⚠️ Those speeds are from a 24-core desktop, so expect a laptop to be several times
slower. A one-line answer took about two minutes there, even on the faster row. If
neither fits, the app says so rather than grinding your machine to a halt.

#### 🧰 What it can and cannot do

**This is a concept, and its tool belt is growing but still limited.** It can
count, filter and group your roll, give you means, medians and quartiles, draw a
histogram, explain the statistics, and run a **ratio study**: median ratio,
weighted mean ratio, COD and PRD, by city, neighbourhood or sale year.

📐 **The ratio study matches the Results tab, to four decimals, on all four
property types.** It scores the model only on sales the model never saw, and it
scores the model and the county on the *same* sales, because anything else
flatters one of them. Done the obvious way, the same question reports a COD of
9.98 instead of 11.12, which flatters the model by more than a full point.
`scripts/check_chat_assistant.py` shows you that comparison.

⚙️ **What counts as acceptable is yours to decide.** The rules differ between
jurisdictions and they change, so the app does not tell you. Enter your own ranges
for level, COD and PRD on the **Settings** tab. The colours on the Results tab and
the assistant's answers judge against those. The boxes start with placeholder numbers
rather than a standard, so replace them before relying on a colour.

🔍 **A small group is kept out of any ranking.** A statistic from a handful of sales
mostly measures the handful, so groups below the line you set on the Settings tab come
with a warning and are listed apart. A city with five sales cannot come out as the
worst in the county.

🛑 **Check its figures on anything more complicated.** A question outside those
tools depends on SQL a small model wrote for itself, and that is where it goes
wrong. It also remembers your earlier questions but not its own answers, so
*"explain that more simply"* will not work on Local. It does not compute PRB yet.

The chat tab lists the questions it has been checked on. To check them yourself:

```bash
uv run python scripts/check_chat_assistant.py                  # needs no model and no key
uv run python scripts/check_chat_assistant.py --engine local   # asks the local model
```

That is evidence from one roll, not a guarantee on yours. As tools are added, this
warning gets shorter.

This is a separate step on purpose. Gradient boosting — the actual subject of
this repository — is a CPU job and needs none of it.

> ℹ️ On Windows you will see `triton not found; flop counting will not work`
> every time PyTorch loads. It is harmless. Triton has no Windows build and
> nothing here counts FLOPs.

---

## 📂 What's in here

```
notebooks/   the walkthrough — 254 cells, committed with its outputs, so you
             can read the whole thing on GitHub without running anything
app/         a Streamlit app: load a Florida county tax roll, generate
             benchmark values, and ask questions about the results
data/        the Lee County roll, compressed. See data/README.md
scripts/     extract_data.py (run this), compress_data.py (maintainer only), and
             check_chat_assistant.py, which checks the chat assistant's answers
```

---

## 🗺️ The data

Real, certified, public: the **2025 Lee County, Florida** tax roll — 557,813
parcels, 123,666 sales, and parcel geometry for all of them.

**🙏 Lee County, Florida gave permission for this data to be used here.** The
records are public, but permission was asked anyway, because asking is the right
way to do this.

⚠️ These records describe real people's homes. Please read
**[data/README.md](data/README.md)** — it covers provenance, what every column
means in appraisal terms, and one important gotcha: **the county's shapefile
ships without a projection file**, and the correct CRS for Lee County is
`EPSG:2237` (NAD83 / Florida West, in **feet**). A different Florida county is a
different zone, and the wrong one silently puts every parcel in the wrong place.

---

## 📐 What you'll actually learn

The walkthrough goes the whole way, including the parts that usually get skipped:

- Loading and joining the **NAL** and **SDF** files every Florida county has
- Feature engineering an appraiser would recognise — effective age, time
  adjustment, quality codes, neighbourhood and market area
- Turning parcel polygons into coordinates the model can use
- **LightGBM, XGBoost and CatBoost** compared head to head, not asserted
- Watching a model **overfit on purpose**, then fixing it with early stopping
- Hyperparameter tuning from manual, to grid search, to **Optuna** — including
  a section on **HPO leakage**, where tuning against your test set quietly
  flatters the result
- **Cross-validation**, and what an odd fold is actually telling you
- **IAAO ratio statistics** — COD, PRD, PRB — not just R²; a board of
  equalization has never asked anyone for an R²
- **SHAP**, to turn "the computer said $412,000" into a line-item receipt, and
  to pull a depreciation curve straight out of the model
- An honest closing section on **what this model can't tell you** — one year of
  unvetted sales, a thin feature set, and why some of the SHAP curves look odd
  because of it

---

## 🚧 Status

Actively being modernized. The environment is built and verified; the notebook
and app are being brought up to current libraries one at a time. Expect rough
edges in the app in particular.

Found something wrong? **Please open an issue.** Corrections are the most
valuable thing anyone can contribute to a teaching repository.

## 📄 License

MIT — see [LICENSE](LICENSE). Use it, teach from it, adapt it for your county.

---

*See also: [88mph-mass-appraisal](https://github.com/Joshua-Data-Wizard/88mph-mass-appraisal)
— the same idea for LLMs, agents and RAG. 🚗💨*

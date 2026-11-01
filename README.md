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
scripts/     extract_data.py (run this) and compress_data.py (maintainer only)
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

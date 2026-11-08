# 🗺️ The Data — Lee County, Florida

> ## 🐬 Start here
>
> The data in this folder is **compressed**. Unpack it once, after `uv sync`:
>
> ```bash
> uv run python scripts/extract_data.py
> ```
>
> Nothing is downloaded — everything you need is already in your clone.
> Takes under a minute and about 1.4 GB of disk.

---

## 🙏 Permission and credit

**Lee County, Florida has given permission for this data to be used in this
repository.** This is a real, certified tax roll from a real county, and the
walkthrough works because they were generous enough to say yes. Thank you.

The underlying records are **public** — Florida's NAL and SDF files are
published annually through the Florida Department of Revenue, and county
parcel geometry is published by the property appraiser's office. Permission
was sought anyway, because asking is the right way to do this.

The MIT license on this repository covers its code and writing. It does not cover these records,
which are not ours to license.

### ⚠️ Public does not mean careless

These records describe **real people's homes**. If you fork this, build on it,
or present from it:

- Don't single out individual parcels or owners in slides, blog posts or
  screenshots.
- Don't join this to anything that re-identifies occupants.
- Remember that an estimate attached to a real address is a statement about a
  real household.

## 🧹 What we removed, and why

The county's files list every parcel's **owner name and mailing address**, and those of any
fiduciary. That is public information: the Department of Revenue publishes it, and anyone can
download it. But nothing in this project uses it, and a public repository should not carry real
people's names further than it has to. So these 14 columns are removed, from the NAL and from the
shapefile's attribute table:

```
OWN_NAME   OWN_ADDR1   OWN_ADDR2   OWN_CITY   OWN_STATE   OWN_ZIPCD   OWN_STATE_DOM
FIDU_NAME  FIDU_ADDR1  FIDU_ADDR2  FIDU_CITY  FIDU_STATE  FIDU_ZIPCD  FIDU_CD
```

**Everything else is exactly as the county published it:** every row, and every other value,
character for character. The property's own address, `PHY_ADDR1`, stays, because it describes the
land and the chat assistant uses it. Removing the columns changed no result in the notebook or the
app, and that was checked rather than assumed: every metric, prediction and table was compared before
and after.

`scripts/compress_data.py` does the removal, and `data_manifest.json` records the checksum of each
file as the county published it. So a copy of the full roll from the
[Department of Revenue's data portal](https://floridarevenue.com/property/dataportal/Pages/default.aspx?path=/property/dataportal/Documents/PTO%20Data%20Portal/Tax%20Roll%20Data%20Files)
can be checked against it, and running the script on it gives these files exactly.

---

## 📦 What is in here

| File | Compressed | Extracted | What it is |
|---|---|---|---|
| `NAL46F202501.csv.gz` | 32 MB | 229 MB | **Name · Address · Legal** — the tax roll. 557,813 parcels × 151 columns, owner names and addresses removed |
| `SDF46F202501.csv.gz` | 3 MB | 18 MB | **Sales Data File** — 123,666 sales × 23 columns |
| `lee_2025par.shp.gz` | 83 MB | 293 MB | parcel **geometry** — 559,141 polygons |
| `lee_2025par.dbf.gz` | 45 MB | 865 MB | the shapefile's attribute table, 105 columns, owner names and addresses removed |
| `lee_2025par.shx.gz` | 2 MB | 4 MB | the shapefile's index |
| `lee_2025par.prj` | *committed plain* | — | the **projection definition** — see the warning below |
| `data_manifest.json` | *committed plain* | — | sizes and SHA-256 checksums, used to verify the extract |

**Naming:** `46` is the Florida DOR county code for Lee County, `F` is the
*final* (certified) roll, and `202501` is the 2025 roll year. Your own county's
files will follow the same pattern with a different number.

### Why compressed?

**GitHub refuses any push containing a file over 100 MB.** Three of these files
are over it, one by 11×. Gzip gets all five comfortably under, because
fixed-width DBF records and comma-separated text repeat themselves enormously:

```
raw total  1,409 MB   ->   committed  ~165 MB
```

You get back exactly the files we packed, the county's less the owner columns, and
`extract_data.py` verifies every one against a SHA-256 checksum.

---

## 🧭 ⚠️ The shapefile has no projection file — and that matters

**The shapefile as supplied by the county is missing its `.prj`.** Without it,
geopandas loads the geometry with `crs = None`, and anything that needs to know
*where on Earth* those coordinates are will fail:

```python
Shape_Data = gpd.read_file("data/lee_2025par.shp")
Shape_Data.crs                 # -> None
Shape_Data.to_crs("EPSG:4326") # -> ValueError: cannot transform naive geometries
```

So **this repository ships a `.prj` we generated** (`scripts/compress_data.py`
writes it), which makes the dataset self-describing for geopandas, QGIS, ArcGIS
and `ogr2ogr` alike. The notebook also declares it explicitly, so the lesson is
visible rather than magic:

```python
Shape_Data = Shape_Data.set_crs("EPSG:2237", allow_override=True)
```

### 🛑 EPSG:2237 is specific to Lee County. Check yours.

`EPSG:2237` is **NAD83 / Florida West (ftUS)** — feet, not metres. Florida has
three State Plane zones, and using the wrong one silently puts every parcel in
the wrong place:

| Zone | EPSG | Roughly |
|---|---|---|
| Florida **East** | 2236 | the Atlantic side — Miami-Dade, Broward, Palm Beach… |
| Florida **West** | **2237** | the Gulf side — **Lee**, Collier, Hillsborough, Pinellas… |
| Florida **North** | 2238 | the panhandle — Leon, Escambia, Duval… |

If you are adapting this to your own county, find its real CRS before you trust
a single coordinate. Your GIS department knows it.

---

## 📋 The columns you actually need

The NAL file has 151 columns and the shapefile 105, once the owner columns are out. The notebook explores them
properly (the Vogons explain a few), but these are the ones the model is built
from:

| Column | Meaning in appraisal terms |
|---|---|
| `PARCEL_ID` | the parcel's unique key — joins NAL, SDF and geometry |
| `DOR_UC` | **DOR Use Code.** `01` is improved single-family residential |
| `SALE_PRC` | sale price — the **target** we are predicting |
| `QUAL_CD` | sale **qualification** code; `01`/`02` are the arm's-length sales |
| `SALE_YR`, `SALE_MO` | when it sold — becomes `Time_to_Sale` |
| `ACT_YR_BLT` | actual year built — becomes `Act_Age` |
| `EFF_YR_BLT` | effective year built, after renovations — becomes `Eff_Age` |
| `TOT_LVG_AREA` | total living area, square feet |
| `LND_SQFOOT` | land area, square feet |
| `IMP_QUAL` | improvement quality, 1 (minimum) to 6 (superior) |
| `NBRHD_CD` | neighbourhood code — the appraiser's market stratification |
| `MKT_AR` | market area — a broader grouping than neighbourhood |
| `CENSUS_BK` | census block |
| `ASMNT_YR` | assessment year, 2025 — used to turn a year built into an age |

⚠️ **DBF truncates field names to 10 characters**, so the shapefile spells some
columns differently from the CSV: `TOT_LVG_AR` in the shapefile is
`TOT_LVG_AREA` in the NAL. Same data, two spellings, depending on which file it
came out of.

### What the model is trained on

After filtering to **improved residential (`DOR_UC == '01'`) with qualified
sales (`QUAL_CD` in `01`, `02`)** and dropping rows missing a feature:

```
557,813 parcels  ->  123,666 sales  ->  25,477 residential qualified  ->  25,348 usable
```

Median sale price in the holdout: **$399,990**.

---

## 🔄 Getting a fresher roll, or your own county

Florida publishes NAL and SDF files annually through the **Department of
Revenue**; parcel geometry comes from the **county property appraiser**. To
point this notebook at a different roll:

1. Drop the new files in `data/`.
2. Update the filename constants near the top of the notebook.
3. **Change the EPSG code** to your county's zone (see the table above).
4. Re-check the sale qualification codes — `01`/`02` is Florida's convention,
   not a universal one.

If you are the maintainer and want to re-pack a new roll for the repository:

```bash
uv run python scripts/compress_data.py
```

That removes the owner columns, rebuilds any archive whose file changed, and writes the `.prj` and
the manifest. It refuses to finish if an archive would exceed GitHub's 100 MB limit, or if an owner
column survives.

---

## 🔖 Provenance

| | |
|---|---|
| **Source** | Lee County, Florida — 2025 certified tax roll (NAL + SDF) and parcel geometry |
| **County DOR code** | 46 |
| **Roll year** | 2025, final/certified |
| **Files dated** | 2026-02-07 |
| **Permission** | ✅ granted by Lee County, Florida |
| **CRS as supplied** | none — no `.prj`. It is **EPSG:2237**, NAD83 / Florida West (ftUS) |
| **Changes we made** | removed the 14 owner name and mailing address columns, added a `.prj`, gzipped the files so they fit on GitHub, and recorded SHA-256 checksums, including each original's |

**Apart from those 14 columns, nothing in these files has been altered, sampled or synthesised.**
Every row, and every other value, is what the county certified.

# arXiv Submission Bundle Manifest

**Paper:** Value Projection Failure in a Near-Cancellation Regime:
A Mechanistic Account of Compositional Generalization Failure in T5-small
**Author:** Troy Teno
**Status:** Bundle verified — see verification notes below.
**Verified:** 2026-06-17 (isolated temp directory, two separate compile modes)

---

## Two Bundle Contexts

This manifest distinguishes two different compilation contexts:

**Local verification bundle (19 files):** Used to confirm the LaTeX source is
self-consistent. Runs `pdflatex + bibtex + pdflatex + pdflatex`. Includes
`bib/references.bib` so BibTeX can regenerate the bibliography from scratch.

**arXiv upload bundle (20 files):** Used for actual submission. arXiv's TeX
processing system does **not** run BibTeX; it expects a pre-generated `main.bbl`
included in the upload tarball. The `.bbl` must be generated locally first, then
included alongside the source. Both `bib/references.bib` and `main.bbl` are
included: the `.bib` for provenance; the `.bbl` because arXiv requires it.

Both bundles were verified in an isolated temp directory (no data/, scripts/,
notes/, figures/*.png, or any repo files outside paper/). Exit 0 in both cases.

---

## Included Files

All packages used (amsmath, amssymb, graphicx, geometry, booktabs, microtype,
natbib, hyperref) are available in the standard TeX Live distribution.
No custom style files, class files, or `.sty` files are required.

### Root

| File | Local bundle | arXiv upload | Role |
|------|:---:|:---:|------|
| `main.tex` | ✓ | ✓ | Root document. Inputs all section files, calls bibliography. |
| `main.bbl` | — | ✓ | Pre-generated bibliography. Required by arXiv (arXiv does not run BibTeX). Generate locally with `bibtex main` before packaging the upload tarball. |

### Sections (12)

| File | Role |
|------|------|
| `sections/00_abstract.tex` | Abstract |
| `sections/01_introduction.tex` | §1 Introduction |
| `sections/02_background.tex` | §2 Background and Related Work |
| `sections/03_born_filter.tex` | §3 The Born Filter Instrument |
| `sections/04_experimental_setup.tex` | §4 Experimental Setup |
| `sections/05_locating_failure.tex` | §5 Results: Locating the Failure |
| `sections/06_explaining_failure.tex` | §6 Results: Explaining the Failure |
| `sections/07_repairing_failure.tex` | §7 Results: Repairing the Failure |
| `sections/08_phase_diagram.tex` | §8 Controlled Geometry Reference Model |
| `sections/09_discussion.tex` | §9 Discussion |

All 10 inputted sections are filled. No TODO-stub sections remain in the compiled
paper. §10 (`10_limitations.tex`) and §11 (`11_future_work.tex`) are not inputted
by `main.tex`; their content is covered by §9.4 and §9.5 respectively. Those files
remain in the repository as inactive scaffold but are not compiled or submitted.

### Bibliography

| File | Local bundle | arXiv upload | Role |
|------|:---:|:---:|------|
| `bib/references.bib` | ✓ | ✓ | BibTeX source. Six entries, all actively cited: lake2018scan, raffel2020t5, geiger2021causal, lo2024contextuality, coecke2010mathematical, abramsky2004categorical. No uncited entries. |
| `main.bbl` | — | ✓ | Generated from `references.bib` by running `bibtex main`. Include in arXiv tarball; do not include local `main.pdf`. |

### Figures (7 PDF files)

| File | Referenced in |
|------|---------------|
| `figures/fig01_failure_rate_by_command_type.pdf` | §1 Fig. 1 (`fig:failure-rate`) |
| `figures/fig02_decoder_routing_summary.pdf` | §5 Fig. 2 (`fig:routing-summary`) |
| `figures/fig03_value_direction_evidence.pdf` | §5 Fig. 3 (`fig:value-direction`) |
| `figures/fig04_near_cancellation.pdf` | §6 Fig. 4 (`fig:near-cancellation`) |
| `figures/fig05_three_regime_phase_diagram.pdf` | §8 Fig. 5 (`fig:phase-diagram`) |
| `figures/fig06_rank1_repair_dose_response.pdf` | §7 Fig. 6 (`fig:dose-response`) |
| `figures/fig07_two_track_near_cancellation.pdf` | §6, §8 Fig. 7 (`fig:two-track`) |

---

## Excluded Files and Exclusion Reasons

### `paper/figures/*.png` — PNG copies of all 7 figures

**Reason:** arXiv accepts PDF figures; the PDF versions are included.
PNG files are redundant for the submission bundle. They are kept in the
repository as raster previews.

### `paper/data/*.csv` — 7 figure source data files

**Reason:** Not required for LaTeX compilation. These are the single
source of truth for plotted values; they belong in the repository provenance
record but are not submitted to arXiv.

| File | Source for |
|------|------------|
| `data/fig01_failure_rates.csv` | fig01 |
| `data/fig02_decoder_routing_summary.csv` | fig02 |
| `data/fig03_value_direction_evidence.csv` | fig03 |
| `data/fig04_near_cancellation.csv` | fig04 |
| `data/fig05_phase_diagram.csv` | fig05 |
| `data/fig06_dose_response.csv` | fig06 |
| `data/fig07_two_track_near_cancellation.csv` | fig07 |

### `paper/scripts/*.py` — 7 figure generation scripts

**Reason:** Not required for LaTeX compilation. These are the figure
production scripts; they belong in the repository for reproducibility
but are not submitted to arXiv.

### `paper/notes/*.md` — 8 working notes files

**Reason:** Internal working documents — claim boundary, figure plans,
production logs, assembly logs, source maps, draft abstracts.
Not part of the LaTeX source; do not belong in the submission bundle.

| File | Contents |
|------|----------|
| `notes/ABSTRACT_v0.1_HOT.md` | Draft abstract (superseded) |
| `notes/ABSTRACT_v0.2_PUBLIC.md` | Draft abstract (superseded) |
| `notes/CLAIM_BOUNDARY.md` | Internal claim discipline document |
| `notes/FIGURE_PLAN.md` | Figure design notes |
| `notes/FIGURE_PRODUCTION_LOG.md` | Figure production log |
| `notes/FIGURE_SOURCE_AUDIT.md` | Figure source audit |
| `notes/PAPER_ASSEMBLY_LOG.md` | Assembly log |
| `notes/PAPER_SOURCE_MAP.md` | Source map |

### `paper/.gitignore` — LaTeX build artifact exclusion rules

**Reason:** Git configuration file. Not part of the LaTeX source.

### LaTeX build artifacts — `main.aux`, `main.blg`, `main.log`, `main.out`, `main.pdf`

**Reason:** Generated by the compile process. Not tracked in git (excluded
by `paper/.gitignore`). arXiv generates its own PDF from source; `main.pdf`
must not be included in the submission tarball.

**Exception — `main.bbl`:** The `.bbl` file is a build artifact but is
**required in the arXiv upload tarball**. arXiv does not run BibTeX; it
expects a pre-generated `.bbl`. Generate it locally (`bibtex main` after
the first `pdflatex` pass), include it in the tarball, and do not rely on
arXiv to produce it. The `.bbl` is currently excluded from git tracking
by `paper/.gitignore`; that is intentional for the repo (the file is
regeneerable), but before packaging the tarball it must be present on disk.

---

## Repository Files Outside `paper/` — All Excluded

| Path | Reason |
|------|--------|
| `DCRP/` | Lab notebooks and design decision records (sandboxes 014–024, lexicon, public readme draft, Geiger literature connection). Internal research process; not submitted. |
| `findings/` | Sealed methods reports for all S-track (S-039–S-062) and G-track (G-039–G-057) experiments. Provenance record for numerical claims; not submitted to arXiv. |
| `essays/` | Six exploratory essays written during the research process. Not the paper; not submitted. |
| `workbench/` | Experiment scripts and proposals for workbench experiments. Code provenance; not submitted. |
| `EXPERIMENT_INDEX.md` | Full experiment index for both tracks. Internal record; not submitted. |
| `FINDINGS.md`, `WHAT_WE_FOUND.md`, `WHAT_WE_FOUND_VOL_II.md` | Research summary documents. Not submitted. |
| `PRIOR_ART.md`, `REPRODUCIBILITY.md`, `THE_SAME_RIVER.md` | Research process and provenance documents. Not submitted. |
| `LESSONS_FROM_REPO_1.md` | Internal lessons document. Not submitted. |
| `Lexicon of the Glass Attention Workbench.md` | Terminology reference. Not submitted. |
| `README.md` | Repository README. Not submitted. |
| `LICENSE` | License file. Not submitted (arXiv handles licensing separately). |
| `requirements.txt` | Python dependencies for workbench scripts. Not submitted. |

---

## Remaining Items Before Final Submission

1. **Stub sections: resolved.** All 10 compiled sections are filled. §10 and §11
   are removed from `main.tex`; their content is in §9.4 and §9.5.

2. **hyperref PDF bookmark warnings: resolved.** All math-in-heading subsections
   wrapped with `\texorpdfstring{}{}`:
   - §1.2: `The Locate $\to$ Explain $\to$ Repair Chain`
   - §7.1: `Rank-1 $W_V$ Correction`
   - §9.1: `The Locate $\to$ Explain $\to$ Repair Chain as Mechanistic Account`
   Full compile (two-pass) produces zero "Token not allowed in PDF string" warnings.
   Upload-bundle compile in isolated temp directory: exit 0, zero warnings.

3. **Uncited bibliography entries: resolved.** `sargsyan2026functoriality` and
   `gavranovic2024fundamental` removed from `references.bib`. Six entries remain,
   all actively cited in text.

4. **arXiv category and metadata.** Not a LaTeX issue; must be entered at arXiv upload:
   recommended category `cs.LG` with cross-list `cs.CL`.

Reading Both Halves of Imvo: the consequences of modernising 1880s isiXhosa
COMS4054A/7062A/7066A NLP Project 2

QUESTION
  Do the isiXhosa and English sections of Imvo Zabantsundu (1884-1894) differ in content once the
  language signal is removed, and how does modernising the 1880s isiXhosa spelling change that answer?

SETUP
  python -m pip install -r requirements.txt          (Python 3.11+)
  Embedding needs a GPU in practice (~1 h on a free Colab T4); everything else runs on a laptop CPU.

DATA
  Unzip TxtLangFiles1870-1894_TB.zip into data/raw/ so that these folders exist:
    data/raw/TxtLangFiles1870-1894_TB/XhTxt
    data/raw/TxtLangFiles1870-1894_TB/EnTxt
  Modern isiXhosa corpora (Vukuzenzele, isiXhosa Wikipedia, OPUS-100 en-xh) download automatically
  from Hugging Face in step 1.

STARTING FROM A CLONE OF THIS BRANCH
  The repository holds code only. The course data and every generated .parquet file are deliberately
  not committed, so a fresh clone cannot run the Colab notebook until the data is rebuilt:

  1. Clone the branch and install the packages (Python 3.11+):
       git clone -b project-2 https://github.com/froggomelo/nlp-project-1.git
       cd nlp-project-1
       python -m pip install -r requirements.txt

  2. Get TxtLangFiles1870-1894_TB.zip from Moodle and extract it into data/raw/ so that this file
     exists:
       data/raw/TxtLangFiles1870-1894_TB/XhTxt/IMVO_1884-11-03_p001.txt
     Windows "Extract All" sometimes creates a doubled folder
     (data/raw/TxtLangFiles1870-1894_TB/TxtLangFiles1870-1894_TB/); if so, move the inner folder up one
     level. The __MACOSX folder can be deleted.

  3. Build the processed data (~4 min; downloads the modern isiXhosa corpora on the first run):
       python src/run_all.py prepare
     Check data/processed/data_card.md: it must say "kept (>= 1000 words in both languages): 511".
     If it does, your data matches the rest of the team's.

  4. Build the Colab bundle:
       python src/make_colab_bundle.py
     It should list data/processed/chunks.parquet and data/processed/normaliser_candidates.parquet
     and report a size of about 26-27 MB. Upload colab/colab_bundle.zip in the notebook (step 3 below).

  Every step in "prepare" is deterministic (seed 13), so rerunning it is safe: it overwrites its
  outputs with identical files and never touches labels/ or data/embeddings/. The one difference
  between machines can come from the modern word list, which is downloaded from Hugging Face at build
  time; if the team needs identical normaliser output, share one colab_bundle.zip instead of each
  person rebuilding it. Do NOT rerun "normalise.py sheet" or "pairs.py sheet" once labelling has
  started: they overwrite the blank labelling sheets.

PIPELINE (run from the project root)
  1. Text preparation (CPU, ~5 min):
       python src/run_all.py prepare
     = prepare_data.py   clean, chunk (60 words), flag boilerplate, split by issue (70/15/15 by year)
       build_lexicon.py  modern isiXhosa word list with per-source counts
       normalise.py      candidate rewrites; tune (if labels exist); apply all / asp / prefix variants
       tool_metrics.py   tokens per word (LaBSE, XLM-R) and out-of-vocabulary rate per variant
       synthetic_check.py  validates the language-removal method on data with a known answer

  2. Hand-labelling (people, ~3-4 h in total; see labels/LABELLING_GUIDE.md):
       python src/normalise.py sheet         -> labels/normaliser_to_label.csv  (save as normaliser_labelled.csv)
       python src/pairs.py sheet              -> labels/pairs_to_label.csv       (save as pairs_labelled.csv;
                                                 needs raw embeddings from step 3)
     After labelling the normaliser sheet, rerun step 1 so the tuned thresholds are used.

  3. Embeddings (GPU):
       python src/make_colab_bundle.py       -> colab/colab_bundle.zip
       open colab/embed_on_colab.ipynb in Colab (browser), upload the bundle, run all cells,
       unzip the downloaded embeddings_all.zip into data/embeddings/.
     On a machine with a GPU instead:
       python src/embed.py --variant raw
       python src/embed.py --variant modern --chunks data/processed/chunks_modern.parquet --reuse raw
       (same for modern_asp and modern_prefix; add --model intfloat/multilingual-e5-base
        --prefix "query: " for the second encoder)

  4. Analysis (CPU, ~15 min per encoder):
       python src/run_all.py analyse
       python src/run_all.py analyse --model-tag multilingual-e5-base
     = pairs.py        mutual-nearest-neighbour pairs per issue; threshold tuned on labelled pairs
       classify.py     section classifier with the language subspace (from translated pairs) removed;
                       k chosen so the validation pair floor is at chance; modern variants reuse raw's
                       reference pairs and k
       compare.py      raw vs modern vs ablations: all / floor / gap, alignment, paired bootstrap CIs
       qualitative.py  topic clusters, isiXhosa-only topics, per-issue divergence vs permutation baseline
     Without labels yet, add --threshold 0.55 for a provisional run.

  Check that everything runs without a GPU (stand-in embeddings; numbers meaningless):
       python src/smoke_test.py   then   python src/smoke_test.py --clean

OUTPUTS
  data/processed/data_card.md            corpus statistics
  results/normaliser/                    thresholds, what changed, likely-name rewrites
  results/tool_metrics/                  fertility and OOV per variant and year
  results/synthetic/fig_synthetic.png    method validation
  results/raw/                           Stage 1: stage1.json, fig_k_curve.png, qualitative.md, fig_clusters.png
  results/compare/                       Stage 3: compare.md, fig_compare.png (central figure)
  results/*__multilingual-e5-base/       the same for the second encoder

KEY DESIGN DECISIONS
  - Splits are by issue, so no issue leaks between train, validation and test.
  - Boilerplate (>= 30% of 5-word sequences seen in >= 5 issues) and language-mismatched chunks are
    flagged and excluded through the `usable` column.
  - Cleaning never changes spelling; spelling changes only in normalise.py, on purpose.
  - Language removal uses only translated pairs, whose content is identical; plain mean-centring is
    reported for comparison because it also removes genuine content differences (synthetic_check.py).
  - Only accuracy above the translated-pair floor counts as a content difference.
  - The normaliser rewrites a word only if the original is unattested in modern text and the rewrite is
    attested; thresholds favour precision (F0.5) because over-correcting authors' spelling is the costly error.
  - Alignment across spelling variants is measured on embedding-free (anchor) pairs to avoid selection bias.

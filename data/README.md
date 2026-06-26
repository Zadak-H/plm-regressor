# Data folder

Place your input CSV files here. The tool accepts **any CSV** — the column names are configurable in the GUI.

---

## Training / regression CSV

Must have one sequence column and one numeric target column. Any additional columns are ignored.

```csv
name,sequence,Tm
WT,MKTIIALSYIFCLVFA,85.3
V12A,MKTIIALSAIFCLVFA,83.1
K5R,MRTIIALSYIFCLVFA,86.7
L17F,MKTIIALSYIFCLVFF,88.0
```

---

## Embed-only CSV (no target needed)

Just sequences. Useful for generating embeddings before a training target is available, or for the **Embed** tab.

```csv
id,sequence
seq1,MKTIIALSYIFCLVFA
seq2,MKTIIALSAIFCLVFA
seq3,MRTIIALSYIFCLVFA
```

---

## Predict CSV

Same format as embed-only — sequences you want predictions for after a model is trained.

```csv
id,sequence
candidate1,MKTIIALSYIFFLVFA
candidate2,MKTIIALSYIYCLVFA
```

---

## Tips

- **Column names**: set them in the GUI dropdowns — no renaming required.
- **Sequences**: standard 20-letter amino-acid alphabet. Non-standard residues (B, U, Z, O, X) are handled by most backends.
- **No size limit** enforced by the tool, but very large datasets (> 10 k sequences) slow down embedding generation significantly — use GPU and a smaller batch size.

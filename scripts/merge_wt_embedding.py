import numpy as np
import sys

# ---- EDIT THESE PATHS ----
TRAIN_NPZ = "embeddings/esm2.npz"
ZEROSHOT_NPZ = "zeroshot_embeds/esm2_650M.npz"
OUTPUT_NPZ = "zeroshot_embeds/esm2_with_wt.npz"

WT_SEQ = """TANPYQRGPDPTESSIEAVRGPFAVAQTTVSRLDASGFGGGTIYYPTDTSQGTFGAVAISPGFTAGQSSIAWLGPRIASQGFVVITIDTISRFDYPDSRGRQLQAALDYLTTDSTVRDRIDPNRMAVMGHSMGGGGALSAAANNPSLKAAIPLQGWHTRKDWSSVRVPTLIVGAELDTIAPVSSHSEAFYNSLPSSLPKAYMELRGASHTVSNTPNTTTAKYSIAWLKRFVDDDTRYEQFLCPAPDDPAISEYRSTCPF""".strip()

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    if "seqs" in data and "embeddings" in data:
        seqs = data["seqs"]
        embeds = data["embeddings"]
    elif "sequences" in data and "embeddings" in data:
        seqs = data["sequences"]
        embeds = data["embeddings"]
    else:
        emb_dict = data["arr_0"].item()
        seqs = list(emb_dict.keys())
        embeds = np.stack([emb_dict[s] for s in seqs], axis=0)

    seqs = [
        s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
        for s in seqs
    ]
    seqs = [s.strip() for s in seqs]

    if embeds.ndim == 3:
        embeds = embeds.mean(axis=1)

    return seqs, embeds

# --- load training NPZ ---
train_seqs, train_embeds = load_npz(TRAIN_NPZ)

if WT_SEQ not in train_seqs:
    raise ValueError("WT not found in training NPZ")

wt_idx = train_seqs.index(WT_SEQ)
wt_vec = train_embeds[wt_idx]

print("Found WT in training NPZ. Dim:", wt_vec.shape)

# --- load zeroshot NPZ ---
zs_seqs, zs_embeds = load_npz(ZEROSHOT_NPZ)

if wt_vec.shape[0] != zs_embeds.shape[1]:
    raise ValueError(
        f"Dimension mismatch: WT dim {wt_vec.shape[0]} vs zeroshot dim {zs_embeds.shape[1]}"
    )

if WT_SEQ in zs_seqs:
    print("WT already exists in zeroshot NPZ. Nothing to do.")
    sys.exit(0)

# --- append ---
new_seqs = zs_seqs + [WT_SEQ]
new_embeds = np.vstack([zs_embeds, wt_vec.reshape(1, -1)])

np.savez_compressed(
    OUTPUT_NPZ,
    sequences=np.array(new_seqs),
    embeddings=new_embeds,
)

print("Saved merged NPZ to:", OUTPUT_NPZ)
print("New shape:", new_embeds.shape)

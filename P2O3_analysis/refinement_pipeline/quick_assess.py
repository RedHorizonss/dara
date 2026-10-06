import json
import pandas as pd

# reload fresh each time — batch is still writing to this file
df = pd.read_csv("batch_results.csv")

def decode_flagged(val):
    if pd.isna(val) or val == "":
        return None
    return json.loads(val)

df["flagged"] = df["flagged"].apply(decode_flagged)
is_flagged = df["flagged"].apply(lambda x: bool(x))

total = len(df)

# 1. how many have >95% impurity (regardless of flagged status)
high_impurity_count = (df["impurity_percentage"] > 90).sum()

# 2. what % of the current dataset is flagged
flagged_count = is_flagged.sum()
flagged_pct = flagged_count / total * 100

# 3. how many usable adjusted compositions right now
success_mask = (
    df["adjusted_formula"].notna()
    & df["p2_o3_ratio"].notna()
    & (df["p2_o3_ratio"] != -1)
    & (df["impurity_percentage"] < 90)
    & (df["adjusted_formula"] != df["original_formula"])
)
success_count = success_mask.sum()

print(f"Processed so far: {total}")
print(f">90% impurity: {high_impurity_count} ({high_impurity_count/total*100:.1f}%)")
print(f"Flagged: {flagged_count} ({flagged_pct:.1f}%)")
print(f"Usable adjusted compositions: {success_count} ({success_count/total*100:.1f}%)")
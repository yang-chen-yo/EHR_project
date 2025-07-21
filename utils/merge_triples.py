# merge_triples.py
import glob, pickle

def main():
    paths = sorted(glob.glob("triples_*.pkl"))
    merged = []
    for p in paths:
        with open(p, "rb") as f:
            merged.extend(pickle.load(f))
    with open("triples_full.pkl", "wb") as f:
        pickle.dump(merged, f)
    print(f"[INFO] 合併完畢，總 triples = {len(merged)}，輸出 triples_full.pkl")

if __name__ == "__main__":
    main()


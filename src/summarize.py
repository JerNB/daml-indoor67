"""Print a table of all runs in results/runs/, best first."""
import glob, json
import pandas as pd

rows = [json.load(open(f)) for f in glob.glob("results/runs/*.json")]
if rows:
    print(pd.DataFrame(rows).sort_values("val_acc", ascending=False).to_string(index=False))
else:
    print("no runs yet")

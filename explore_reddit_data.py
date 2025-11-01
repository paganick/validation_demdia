import pickle
import pandas as pd

# --- CONFIG ---
pickle_file = "reddit_data.pkl"   # input pickle file
csv_file = "reddit_data.csv"      # output csv file
# --------------

# Load pickle
with open(pickle_file, "rb") as f:
    data = pickle.load(f)

# If it's a DataFrame, just save to CSV
if isinstance(data, pd.DataFrame):
    data.to_csv(csv_file, index=False)
    print(f"✅ Pickle DataFrame exported to {csv_file}")
else:
    # Try converting other Python objects (like dicts or lists) into DataFrame
    try:
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        print(f"✅ Pickle object converted to DataFrame and exported to {csv_file}")
    except Exception as e:
        print("❌ Could not convert pickle object to DataFrame.")
        print("Type of object:", type(data))
        print("Error:", e)
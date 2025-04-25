import pandas as pd

# Load the CSV file
df = pd.read_csv("simulation_results10/meta-llama/Llama-3.1-8B__noft__ctx1__style0__OPPU__random_response_validation_data.csv", header=None, names=["text", "label"])

# Count occurrences of each label
label_counts = df["label"].value_counts()

print(label_counts)
import os
import yaml

input_folder = "."  # change this to your actual path

for filename in os.listdir(input_folder):
    if filename.startswith("deepseekR1") and not filename.startswith("mistral"):
        filepath = os.path.join(input_folder, filename)

        with open(filepath, "r") as f:
            config = yaml.safe_load(f)

        # Modify config
        config["model"] = "mistralai/Mistral-7B-v0.1"

        # Build new filename
        new_filename = "mistral" + filename.replace("deepseekR1", "")
        new_filepath = os.path.join(input_folder, new_filename)

        with open(new_filepath, "w") as f:
            yaml.dump(config, f)

        print(f"✅ Created {new_filename}")

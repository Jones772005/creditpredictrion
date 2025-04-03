import pandas as pd

# Load dataset with proper encoding and error handling
file_path = r"D:\\Jones\\credit\\dataset\\application_record.csv"  # Use raw string (r"") or double backslashes
df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)

# Fill missing values in critical columns (optional)
df.fillna("", inplace=True)

# Add the "approved" column based on conditions
df["approved"] = (
    (df["AMT_INCOME_TOTAL"] >= 200000) &  # Income ≥ 200K
    ((df["DAYS_BIRTH"] / -365) >= 18) &  # Age ≥ 18
    ((df["DAYS_EMPLOYED"] / -365) >= 2) &  # Employment ≥ 2 years
    (df["CNT_CHILDREN"] < 4)  # Less than 4 children
).astype(int)  # Convert Boolean to 0/1

# Save the updated dataset
output_file = r"D:\\Jones\\credit\\dataset\\record.csv"
df.to_csv(output_file, index=False)

print(f"Updated dataset saved as {output_file}")

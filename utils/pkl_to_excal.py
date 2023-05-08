import pandas as pd
import pickle

# Load the pkl file
with open('./data/vadere3.pkl', 'rb') as f:
    data = pickle.load(f)

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
df.to_excel('./data/vadere3.xlsx', index=False)

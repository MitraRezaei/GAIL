import pandas as pd

m = pd.read_json(r"vadere_scene_1.json",orient="records")
m.to_csv("./data/obstacle.csv")

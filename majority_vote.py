import pandas as pd
import random

test_path = "annotations/query_frame_annotations_val.csv"

df = pd.read_csv(test_path)
df_maj = pd.DataFrame(columns=df.columns)
imageURL = df['imageURL'].drop_duplicates()

for i, url in enumerate(imageURL):
    df_rev = df[ df['imageURL'] == url ]
    digits = {}
    candidate = []
    for j in df_rev['Relevance']:
        digits[j] = digits.get(j, 0) + 1
        if digits[j] > df_rev['Relevance'].size / 2:
            majority = j
            break
    if max(digits.values()) < df_rev['Relevance'].size / 2:
        for j in df_rev['Relevance']:
            if digits[j] == 2:
                candidate.append(j)
        majority = random.choice(candidate)
    df_maj = pd.concat([df_maj, df_rev[ df_rev['Relevance'] == majority] ], ignore_index=True )
df_maj.drop_duplicates(subset='imageURL', inplace=True)
df_maj.to_csv("annotations/query_frame_annotations_val_major.csv")
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Ensure your backend has the fresh ultimate_items.csv renamed to items.csv
CSV_PATH = 'items.csv' 
NPY_PATH = 'item_embeddings.npy'

print("Loading ML Matrix and Perfect Taxonomy...")
df = pd.read_csv(CSV_PATH)
embeddings = np.load(NPY_PATH)

def get_meal_completion_recs(item_id, top_n=5):
    try:
        # 1. Type-Safe Target Extraction
        idx = df[df['item_id'] == int(item_id)].index[0]
    except IndexError:
        print(f"Error: ID {item_id} not found.")
        return []

    target_vec = embeddings[idx].reshape(1, -1)
    target_cat = df.iloc[idx]['category']
    target_cuisine = df.iloc[idx]['cuisine_type']
    target_res_id = df.iloc[idx]['restaurant_id']
    target_name = str(df.iloc[idx]['name']).lower()

    # 2. Base ML Relevance 
    sim_scores = cosine_similarity(target_vec, embeddings).flatten()
    boosted_df = df.copy()
    boosted_df['similarity'] = sim_scores
    
    # 3. THE RESTAURANT LOCK
    # Immediately drop everything from other restaurants to prevent cart crashes
    boosted_df = boosted_df[boosted_df['restaurant_id'] == target_res_id]

    # 4. THE CUISINE SHIELD
    # Cross-cuisine mixing is heavily penalized. Drinks and Desserts are universal.
    cross_cuisine_mask = (boosted_df['cuisine_type'] != target_cuisine) & (~boosted_df['category'].isin(['Drink', 'Dessert']))
    boosted_df.loc[cross_cuisine_mask, 'similarity'] *= 0.1

    # 5. STRUCTURAL TAXONOMY MULTIPLIERS (The Engine)
    
    if target_cat == 'Wet Curry':
        boosted_df.loc[boosted_df['category'].isin(['Wet Curry', 'Dry Main']), 'similarity'] *= 0.1 # Nuke other mains
        boosted_df.loc[boosted_df['category'] == 'Bread', 'similarity'] *= 5.0 # Demand bread
        boosted_df.loc[boosted_df['category'] == 'Starter', 'similarity'] *= 2.0
        boosted_df.loc[boosted_df['category'] == 'Drink', 'similarity'] *= 1.5

    elif target_cat == 'Dry Main': # Biryani, Noodles, Thali
        boosted_df.loc[boosted_df['category'].isin(['Wet Curry', 'Dry Main']), 'similarity'] *= 0.1
        boosted_df.loc[boosted_df['category'] == 'Side', 'similarity'] *= 4.0 # Demand Raita, Papad, or Sauces
        boosted_df.loc[boosted_df['category'] == 'Starter', 'similarity'] *= 2.0
        boosted_df.loc[boosted_df['category'] == 'Drink', 'similarity'] *= 1.5

    elif target_cat == 'Fast Food Main': # Burgers, Pizzas
        boosted_df.loc[boosted_df['category'] == 'Fast Food Main', 'similarity'] *= 0.1
        boosted_df.loc[boosted_df['category'] == 'Side', 'similarity'] *= 5.0 # Demand Fries/Wedges
        boosted_df.loc[boosted_df['category'] == 'Drink', 'similarity'] *= 3.0

    elif target_cat == 'Bread':
        boosted_df.loc[boosted_df['category'] == 'Bread', 'similarity'] *= 0.1
        boosted_df.loc[boosted_df['category'] == 'Wet Curry', 'similarity'] *= 5.0 # Hunt for Curries
        boosted_df.loc[boosted_df['category'] == 'Dry Main', 'similarity'] *= 0.1 # Avoid Biryani
        boosted_df.loc[boosted_df['category'] == 'Starter', 'similarity'] *= 1.5

    elif target_cat == 'Starter':
        boosted_df.loc[boosted_df['category'] == 'Starter', 'similarity'] *= 0.1
        boosted_df.loc[boosted_df['category'].isin(['Wet Curry', 'Dry Main', 'Fast Food Main']), 'similarity'] *= 3.0

    elif target_cat == 'Dessert':
        boosted_df.loc[~boosted_df['category'].isin(['Dessert', 'Drink']), 'similarity'] *= 0.01

    # Soft Dessert Kill-Switch for Mains
    if target_cat in ['Wet Curry', 'Dry Main', 'Fast Food Main', 'Bread']:
        boosted_df.loc[boosted_df['category'] == 'Dessert', 'similarity'] *= 0.4

    # Sort mathematically
    boosted_df = boosted_df.sort_values(by='similarity', ascending=False)
    
    # 6. BULLETPROOF DEDUPLICATION
    final_recs = []
    
    # Pre-seed the blacklist with the anchor item's clean name
    seen_names = {target_name.split('(')[0].strip()}
    
    for _, row in boosted_df.iterrows():
        clean_name = str(row['name']).split('(')[0].strip().lower()
        
        # Safe float-to-int comparison
        if int(float(row['item_id'])) == int(float(item_id)) or clean_name in seen_names:
            continue
            
        final_recs.append(row.to_dict())
        seen_names.add(clean_name)
        
        if len(final_recs) == top_n:
            break
            
    return final_recs
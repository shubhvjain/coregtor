import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

def gene_frequency_patterns(df, min_support=0.04):
    # 1. Prepare the data
    transactions = df['sources'].str.split(';').tolist()

    # 2. One-hot encode the transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    encoded_df = pd.DataFrame(te_ary, columns=te.columns_)

    # 3. Find frequent itemsets
    frequent_itemsets = fpgrowth(encoded_df, min_support=min_support, use_colnames=True)

    # Safety Fix: Handle completely empty itemsets immediately
    if frequent_itemsets.empty:
        return df.iloc[0:0], pd.DataFrame(columns=['source', 'support_size', 'common_targets'])

    # Only keep itemsets containing 2 or more genes (Non-singletons)
    frequent_pairs_groups = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) >= 2)].copy()
    
    if frequent_pairs_groups.empty:
        return df.iloc[0:0], pd.DataFrame(columns=['source', 'support_size', 'common_targets'])

    # ===  MAXIMAL FREQUENT ITEMSETS FILTERING ===
    # Sort by itemset length descending to evaluate largest combinations first
    frequent_pairs_groups['length'] = frequent_pairs_groups['itemsets'].apply(len)
    frequent_pairs_groups = frequent_pairs_groups.sort_values(by='length', ascending=False)
    
    maximal_itemsets = []
    for itemset in frequent_pairs_groups['itemsets']:
        # If this itemset is a subset of any already-saved maximal itemset, skip it
        if not any(itemset.issubset(max_set) for max_set in maximal_itemsets):
            maximal_itemsets.append(itemset)
            
    # Filter the original DataFrame groups down to just the maximal groups
    frequent_pairs_groups = frequent_pairs_groups[frequent_pairs_groups['itemsets'].isin(maximal_itemsets)]
    # =============================================================

    # 4. Extract the unique list/sets of patterns for filtering
    frequent_sets_list = frequent_pairs_groups['itemsets'].tolist()

    # 5. Filter the original DataFrame (Cluster Filtering)
    # Strict Fix: Keep a row ONLY if a complete frequent pattern is present inside it
    def has_frequent_combination(source_str):
        sources_set = set(source_str.split(';'))
        return any(itemset.issubset(sources_set) for itemset in frequent_sets_list)

    filtered_df = df[df['sources'].apply(has_frequent_combination)].copy()

    # 6. Build the frequent_summary [source | support_size | common_targets]
    summary_data = []
    
    for _, row in frequent_pairs_groups.iterrows():
        itemset = row['itemsets']
        source_name = ";".join(sorted(list(itemset)))
        
        # Find which rows contain ALL genes in this itemset
        matches = df[df['sources'].apply(lambda x: itemset.issubset(set(x.split(';'))))]
        
        support_size = len(matches)
        common_targets = ";".join(matches['target'].unique())
        
        summary_data.append({
            'source': source_name,
            'support_size': support_size,
            'common_targets': common_targets
        })
        
    frequent_summary = pd.DataFrame(summary_data)
    frequent_summary = frequent_summary.sort_values(by='support_size', ascending=False).reset_index(drop=True)

    return filtered_df[["cluster_uid","target","sources","n_source"]], frequent_summary
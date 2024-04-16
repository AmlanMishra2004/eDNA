# Reads in the results file
# graphs different metrics using micro accuracy as color

import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import utils


"""
NOTE: keep patience and min_pct_improvement at 5, 1 for purposes of comparison

Results of the first arch search, 2427 models explored:
- all other val metrics could be perfect and you could still get as low as 90% micro val accuracy

- 1 to 2 layers, BUT was affected by bias: more layers = more chance to get an invalid combination of hyperparameters, so many fewer 4-layer models than 1-layer models were evaluated. Regardless, as pie graphs go from all models to higher performing models, the percent of 1 and 2 layer models increased, while 3 and 4 decreased.
- dropout of 0-0.6 (the range of given values) was all high performing
- stride was 1 80% of the time for highest models on layer 1, rarely 2, never 3. On layer 2, it was 50/50 stride 1 or 2.
- kernel size was mostly 5-9, sometimes 11, not too often 3
- pooling kernel size ranged across all given values, 0-3
- output channels were mostly large, 128 and 256, 
- batch size was mostly 16 or 32, but 64 was not uncommon

next steps:
- raise conv channels range and make more granular, [5, 6, 7, 8, 9]
"""

overall_df = pd.read_csv('datasets/v4_combined_reference_sequences.csv', sep=';')
overall_df = utils.remove_species_with_too_few_sequences(
            overall_df, 'species_cat', 2, False
        )
overall_df['seq_length'] = overall_df['seq'].str.len()
print(f"MEAN: {overall_df['seq_length'].mean()}")

plt.figure(figsize=(10,6))
overall_df['seq_length'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Number of Bases in Sequence')
plt.ylabel('Frequency')
plt.title('Sequence Lengths for Species with at Least 2 Sequences') # for Species with >2 Sequences
plt.show()
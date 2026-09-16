import numpy as np
import pandas as pd

data_suserup = pd.read_csv('suserup_data_clean.csv')
data_dalby = pd.read_csv('dalby_data_clean.csv')
data_biskopstorp = pd.read_csv('biskopstorp_data_clean.csv')
data_ukr = pd.read_csv('UKR_data_clean.csv')

print('years of Suserup data:', data_suserup['year'].unique())
print('Species in Suserup data:', data_suserup['entext'].unique())
n = len(data_suserup['entext'].unique())
years = data_suserup['year'].unique()
specie_names = data_suserup['entext'].value_counts().nlargest(n).index.tolist()
for i in range(len(years)):
    for j in range(n):
        data_year = data_suserup[data_suserup['year'] == years[i]]
        df_specie = data_year[data_year['entext'] == specie_names[j]]
        print(f'number of {specie_names[j]} trees in year', years[i], ':', len(df_specie))

# print('years of Dalby data:', data_dalby['year'].unique())
# print('Species in Dalby data:', data_dalby['entext'].unique())
# n = len(data_dalby['entext'].unique())
# years = data_dalby['year'].unique()
# specie_names = data_dalby['entext'].value_counts().nlargest(n).index.tolist()
# for i in range(len(years)):
#     for j in range(n):
#         data_year = data_dalby[data_dalby['year'] == years[i]]
#         df_specie = data_year[data_year['entext'] == specie_names[j]]
#         print(f'number of {specie_names[j]} trees in year', years[i], ':', len(df_specie))

# print('years of Biskopstorp data:', data_biskopstorp['year'].unique())
# print('Species in Biskopstorp data:', data_biskopstorp['entext'].unique())
# n = len(data_biskopstorp['entext'].unique())
# years = data_biskopstorp['year'].unique()
# specie_names = data_biskopstorp['entext'].value_counts().nlargest(n).index.tolist()
# for i in range(len(years)):
#     for j in range(n):
#         data_year = data_biskopstorp[data_biskopstorp['year'] == years[i]]
#         df_specie = data_year[data_year['entext'] == specie_names[j]]
#         print(f'number of {specie_names[j]} trees in year', years[i], ':', len(df_specie))

# print('years of Ukraine data:', data_ukr['year'].unique())
# print('Species in Ukraine data:', data_ukr['entext'].unique())
# n = len(data_ukr['entext'].unique())
# years = data_ukr['year'].unique()
# specie_names = data_ukr['entext'].value_counts().nlargest(n).index.tolist()
# for i in range(len(years)):
#     for j in range(n):
#         data_year = data_ukr[data_ukr['year'] == years[i]]
#         df_specie = data_year[data_year['entext'] == specie_names[j]]
#         print(f'number of {specie_names[j]} trees in year', years[i], ':', len(df_specie))
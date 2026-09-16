import numpy as np
import pandas as pd

# Load data 
data_coords = pd.read_excel('./Biskopstorp/Holkåsen points data.xls')
data_trees_1996 = pd.read_excel('./Biskopstorp/Reinventory Holkåsen Copy of Copy of dead alive 96 and 06.xls', sheet_name='Inventory 1996')
data_trees_2006 = pd.read_excel('./Biskopstorp/Reinventory Holkåsen Copy of Copy of dead alive 96 and 06.xls', sheet_name='Inventory 2006')

# Create DataFrames for each year
df_1996 = pd.DataFrame({
    'N': data_trees_1996['N'],
    'species': data_trees_1996['Trsl'],
    'year': 1996
})

df_2006 = pd.DataFrame({
    'N': data_trees_2006['N'],
    'species': data_trees_2006['sp'],
    'year': 2006
})

# Combine the two years
output_df = pd.concat([df_1996, df_2006], ignore_index=True)

# Add coordinates by merging with data_coords
coords_mapping = data_coords[['N', 'Eo(mm)', 'No(mm)']].rename(columns={
    'Eo(mm)': 'x',
    'No(mm)': 'y'
})

# Merge coordinates with the combined dataset
output_df = output_df.merge(coords_mapping, on='N', how='left')

# Mapping dictionary
tree_species = {
    'B' : 'Fagus sylvatica',
    'B  15 N eller S-kolla upp': 'Fagus sylvatica',
    'B  16 N eller S-kolla upp': 'Fagus sylvatica',
    'B 1': 'Fagus sylvatica',
    'B 2': 'Fagus sylvatica',
    'B dubbelst fr 1,2 m-B1+B2 nedan': 'Fagus sylvatica',
'B i 570' : 'Fagus sylvatica',
'B("sidogren")' : 'Fagus sylvatica',
'B((hst))': 'Fagus sylvatica',
'B(delvis hst)': 'Fagus sylvatica',
 'B(h)st': 'Fagus sylvatica',
 'B(h)stu' : 'Fagus sylvatica',
 'B(hst)' : 'Fagus sylvatica',
 'B(hst)=987': 'Fagus sylvatica',
 'B(hst)-rätt läge se997' : 'Fagus sylvatica',
 'B(trol)stu' : 'Fagus sylvatica',
 'B(trol)stu-totalt ndbr': 'Fagus sylvatica',
 'B/G(9/1) föryngring start mot N' : 'Fagus sylvatica',
 'B½död' : 'Fagus sylvatica',
 'Bdubtopp/toppbr' : 'Fagus sylvatica',
 'Bföryngring': 'Fagus sylvatica',
'Bhst' : 'Fagus sylvatica',
'Bhst ' : 'Fagus sylvatica',
'Bhst ovan N' : 'Fagus sylvatica',
'Bhst ovan S': 'Fagus sylvatica',
 'Bhst/½rotvälta': 'Fagus sylvatica',
 'Bhst/toppbrott' : 'Fagus sylvatica',
'Bhst+dubst' : 'Fagus sylvatica',
'Bhstdubst, fr 1,5m' : 'Fagus sylvatica',
'Bhst-mätt tidigare': 'Fagus sylvatica',
'B-skadad': 'Fagus sylvatica',
'Bstu': 'Fagus sylvatica',
'Bstu jättetickor': 'Fagus sylvatica',
'Bstubbe' : 'Fagus sylvatica',
'Bj' : 'Betula spp.',
'Bj(glas)' : 'Betula spp.',
'Bjstu': 'Betula spp.',
'Bjhst': 'Betula spp.',
'Bjstu m låga': 'Betula spp.',
'Brakved' : 'Frangula alnus',
'Ek' : 'Quercus robur',
'Ek½död': 'Quercus robur',
'En' : 'Juniperus communis',
'Jc' :'Juniperus communis',
'rönn' :'Sorbus aucuparia',
'rönn ':'Sorbus aucuparia',
'rönnstubbe':'Sorbus aucuparia',
'Rönn':'Sorbus aucuparia',
'Rowan':'Sorbus aucuparia',
'rowan':'Sorbus aucuparia',
'T': 'Tilia',
'Tstu': 'Tilia',
'Tstubbe': 'Tilia',
'abies alba': 'Abies alba',
'Aa': 'Abies alba',
'Pa': 'Picea albies',
'pa/fs': 'Picea albies',
'dubbelG gemensamma rötter': 'Picea',
'G': 'Picea',
'G(½rotvälta lev)': 'Picea',
'G(dubst fr 5dm)': 'Picea',
'G(hoftat)': 'Picea',
'G(trol redan inmätt)': 'Picea',
'G+G': 'Picea',
'G=slut+fallri låga756': 'Picea',
'G-3.stammig fr 3-4dm höjd': 'Picea',
'Gföryng 0-10Ö, 5-35S': 'Picea',
'Gföryngring': 'Picea',
'Gföryngring inom/mellan pkt 1243-39-40-46-(41)-43 något åt 47': 'Picea',
'Gföryngring runt pkt;7V-7Ö-20S-20N, 1m höga':'Picea',
'Ghst':'Picea',
'granföryngring-mitt N-S +/-3dm - Ö-V +/-7dm': 'Picea',
'Gstu': 'Picea',
'Gstu m stam': 'Picea',
'Gstu(med avkvistad låga)': 'Picea',
'Gstubbe':'Picea',
'klibbal': 'Alnus glutinosa',
    'Fg': 'Fagus sylvatica',
    'Qr': 'Quercus robur',
    'Ps': 'Pinus sylvestris',
    'Ab': 'Abies alba',
    'Aa': 'Abies alba',
    'Bp': 'Betula spp.',
'Fs': 'Fagus sylvatica'
}

output_df['N'] = output_df['N'].astype(str).str.strip()
output_df['N'] = output_df['N'].str.replace(r'\.0$', '', regex=True)

# Instead of dropping all duplicates, handle them by year
output_df_1996 = output_df[output_df['year'] == 1996].drop_duplicates(subset=['N'], keep='first')
output_df_2006 = output_df[output_df['year'] == 2006].drop_duplicates(subset=['N'], keep='first')
output_df = pd.concat([output_df_1996, output_df_2006], ignore_index=True)

# Add new column with full names
output_df['entext'] = output_df['species'].map(lambda x: tree_species.get(x, 'None'))

# Reorder columns to match desired format
output_df = output_df[['N', 'x', 'y', 'year', 'species', 'entext']]

#Cleaning up some rows with missing data etc
output_df = output_df[~output_df['entext'].isin(['None', None])].reset_index(drop=True)
output_df = output_df.dropna(subset=['x', 'y']).reset_index(drop=True)

# Make x and y from mm to m
output_df['x'] = output_df['x'] / 1000
output_df['y'] = output_df['y'] / 1000

# Save to CSV
output_df.to_csv('biskopstorp_data_clean.csv', index=False)
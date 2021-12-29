# ECES 641 Grad Project
# Phil Huddy
# Raymond Yung

import numpy as np
import pandas as pd
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


# read in the aligned sequences
aligned_sequences = []
for seq in SeqIO.parse("output.mafft", "fasta"):
    aligned_sequences.append(seq)

# get rid of reference sequence
aligned_sequences.pop(0)

# load in ISM annotation info
ism_annotation = pd.read_csv('ISM_annotation.txt', sep=',')

# create an empty array to store ISMs
ism_data = []

# generate the ISMs by grabbing the nucleotide at each Ref position in ISM_annotation.txt
for aligned_seq in aligned_sequences:
    cur_ism = ''
    for pos in ism_annotation['Ref position']:
        cur_ism += aligned_seq.seq[(pos-1)]

    ism_data.append([aligned_seq.name, cur_ism])

isms = pd.DataFrame(ism_data, columns=["name", "seq"])

# display subtypes
print(isms)

# calculate percentages of ambiguous bases
data_length = len(isms['name'])
for seq in range(data_length):
    n_count = isms['seq'][seq].count('n');
    seq_length = len(isms['seq'][seq])
    print(isms['name'][seq],'has', n_count, 'ambigious bases, which is', n_count/seq_length*100, 'of its total sequence')


#%%
import pandas as pd

# gets region where the ISM is found most abundant
def get_most_abundant_region_by_ism(df, ism):
    ism = ism.upper()
    # filter out unwanted columns and rows that dont match the ISM
    filtered_isms = df.loc[df['ISM'] == ism, ['country/region', 'ISM']]
    return filtered_isms['country/region'].mode() # return the region that appears the most

# read in data
ism_df = pd.read_excel('ISM_df_with_correction.xlsx')

# cleaned ISMs
isms = ['ttaacttcggtccgcaccctagggctgctcggg',
        'ttaacttcggtccgcaccctagggcggctcggg',
        'tcaacttcggtccgcaccctagggctgctcggg',
        'tcaacttcggtccgcaccctagggcggctcggg']

# iterate through all ISMs and print out the most abundant region(s)
for ism in isms:
    ism_regions = get_most_abundant_region_by_ism(ism_df, ism)
    modes = ", ".join(ism_regions.array)
    print("The most abundant region(s) for ISM " + ism + " is/are " + modes)

#%%
import pandas as pd
from datetime import datetime

# gets the ISM that was most abundant in California on a specific date
def get_most_abundant_ism_by_date(df, d):
    # filter out unwanted columns and all entries not in california on the specified date
    filtered_isms = df.loc[(df['date'] == d) & (df['division'] == 'California'), ['ISM', 'division', 'date']]
    return filtered_isms['ISM'].mode() # return the ISM that appears the most

# manually create dates
dates = [datetime(2020, 5, 19),
         datetime(2020, 5, 28),
         datetime(2020, 6, 9),
         datetime(2020, 6, 30),
         datetime(2020, 7, 1)]

ism_df = pd.read_excel('ISM_df_with_correction.xlsx')

for d in dates:
    d_seqs = get_most_abundant_ism_by_date(ism_df, d)
    modes = ", ".join(d_seqs.array)
    print("The most abundant ISM(s) in California on " + str(d) + " is/are " + modes)


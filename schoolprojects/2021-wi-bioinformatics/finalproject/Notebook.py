#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ECES 641 Grad Project
# Phil Huddy
# Raymond Yung

import numpy as np
import pandas as pd
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

PHIL_EMAIL = "pdh46@drexel.edu"
RAYMOND_EMAIL = ""

Entrez.email = PHIL_EMAIL

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


# ## What subtype is each sample assigned to?

# In[2]:


print(isms)


# # How is the quality of the ISMs, do you see any ambiguous bases? If so, why is that?  
# 
# - Amgbigious Bases are defined as bases with missing protein markers in them. As Zhao et Al explains, it is the ambiguities in reported sequence data. 
#     - **"Bases like N and - represent a fully amigous site and gap respectivel are substantially less informative."**
#          - -Zhao et Al
#          
# - Yes we do see ambigious bases. For the most part, the quality of our ISM's are pretty good even though there are a few gaps given by the n bases shown in our sequences. As shown below, the ambiguity represnts at most, 12.12% of our total sequences. The wastewater papers in which these samples are associated with states that the 
#     - "SARS-CoV-2 was not strongly correlated with RT-qPCR genome copy quantification, which is **likely due to the variability intruduced by different extraction methods**." 
#     - Furthermore, "Only samples with RT-qPCR CT-values <33 (~25 gc/uL) yielded complete consensus genomes". 

# In[3]:


data_length = len(isms['name'])
for seq in range(data_length):
    n_count = isms['seq'][seq].count('n');
    seq_length = len(isms['seq'][seq])
    print(isms['name'][seq],'has', n_count, 'ambigious bases, which is', n_count/seq_length*100, 'of its total sequence')


# Can you find the consensus ISM of these 7 ISMs?
# - "Aside from the from ambigious bases are any of the other positions matching? And can you replace the ambigious bases with those bases based off that consesus" 

# In[4]:


def split(word):
    return np.asarray([char for char in word])


# In[5]:


isms_modified = [];

for seq in range(data_length):
    isms_modified.append(split(isms['seq'][seq]))
    
isms_modified = np.asarray(isms_modified)


# In[6]:


import collections as collections
isms_modified = np.transpose(isms_modified) #Transpose so we can view columns instead. Easier to iterate imo
[L, W] = np.shape(isms_modified) #Can iterate this L wise


# In[ ]:





# In[7]:


base_list = list('tgcan');
hits = pd.DataFrame(columns = base_list, index = None)

for i in range(L):
    collected = collections.Counter(isms_modified[i])
    hit_array = np.zeros(len(base_list))
    for base, count in collected.items():
        index = base_list.index(base)
        hit_array[index] = count
    hit_array = pd.DataFrame([hit_array], columns = base_list, index = [i])
    hits = hits.append(hit_array)
    
print(hits)


# ## Explanation for what's going on here: 
# 
# I transposd the sequences so that we can view all the bases per position and counted the number of repeated hits per position. Now we can view the ambiguity by seeing all the unique bases that only have 1 consensus base. We can create a mediocre confidence value by also dividing the total value of the consesus at the posisition by the total number of sequences (in our case 7).    
#   
# Building onto this, we can do this for all the possible bases at that position, but really we only want a confidence level that's greater that 70% in my opinion. 

# In[8]:


def confidence(lvl, df):
    hits_confidence = df.copy()
    index = np.where(hits_confidence.div(7) >= lvl)
    row, col = np.shape(index)

    consensus = []
    for i in range(col):
        base = hits_confidence.columns[index[1][i]]
        if base != 'n':
            position = index[0][i]
            count = hits_confidence.iloc[index[0][i]][index[1][i]]
            ncount = hits_confidence.iloc[index[0][i]][4]
            confidence = count/(7-ncount)
            consensus.append([base, position, count, ncount, confidence])
        
    consensus = pd.DataFrame(consensus, columns=["base", "position","count","ncount","confidence"])
    return consensus


# In[9]:


test = confidence(.1, hits)
print(test.to_string(index=False))


# In[10]:


from itertools import groupby
from operator import itemgetter

def convert_to_sequence(df_mat):
    position = df_mat['position'].values.flatten()
    con_index = []
    for k, g in groupby(enumerate(position), lambda x:x[0]-x[1]):
        con_index.append(list(map(itemgetter(1), g)))
    consensus = [];
    confidence = [1];
    test = np.concatenate(con_index)
    
    if len(np.unique(test)) > np.max(position):
        con_index = np.unique(test)
        b_name = []
        for i in con_index:
            p_count = np.where(df_mat['position']==i)[0]
            
            if len(p_count) == 1:
                if len(b_name) <= 1:
                    b_name.append(df_mat['base'][int(p_count)])
                    b_name = ''.join(b_name)
                    confidence[0] = confidence[0]*df_mat['confidence'][int(p_count)]
                    
                else:
                    for i in range(len(b_name)):
                        b_name[i] = [b_name[i], df_mat['base'][int(p_count)]]
                        #print(confidence[i])
                        confidence[i] = confidence[i]*df_mat['confidence'][int(p_count)]
                        b_name[i] = ''.join(b_name[i])
            else:
                n_name = []
                con_1 = []
                for b in range(len(b_name)):
                    for p in p_count:
                        t_con = confidence[b]*df_mat['confidence'][p]
                        temp = [b_name[b], df_mat['base'][p]] 
                        n_name.append(''.join(temp))
                        con_1.append(t_con)
                b_name = n_name
                confidence = con_1
       
        return(b_name, confidence)
    
    
    else:
        for p in con_index:
            base_name = [];
            con = [];
            print(p)
            for b_num in p:
                cur_idx = np.where(df_mat['position']==b_num)[0]
                for idx in cur_idx:
                    base_name.append((df_mat['base'][idx]))
                    con.append((df_mat['confidence'][idx]))
            consensus.append(''.join(base_name))
        
        return(consensus, con_index)


# In[11]:


thing = convert_to_sequence(test)
for num in range(len(thing[0])):
    print(thing[0][num], thing[1][num])


# In[12]:


temp = np.argsort(thing[1])
a = np.asarray(thing[0])
b = np.asarray(thing[1])


# In[13]:


new_thing = a[temp[::-1]]
new_thing_val = b[temp[::-1]]
thing =(a,b)


# In[14]:


for i in range(len(new_thing)):
    print(new_thing[i], new_thing_val[i])


# In[15]:


def lev_string(word1, word2):
    #Get sizes of words... Will need to create matrix
    #Making sure that s1 is the largest
    if len(word1) >= len(word2):
        large = len(word1); small = len(word2)
        l_word = word1; s_word = word2;
    else:
        large = len(word2); small = len(word1)
        l_word = word2; s_word = word1;
        
    #For Consistency... Keep Largest on Horizontal and Smallest on Vertical
    distance = np.zeros((small+1, large+1))
    distance[0,:] = np.array([i for i in range(large+1)])
    distance[:,0] = np.array([i for i in range(small+1)])
    
    for col in range(1, large+1):
        for row in range(1, small+1):
            #Only add 1 if the characters at word1[i] not equal word2[j]
            if (l_word[col-1] == s_word[row-1]):
                distance[row, col] = distance[row-1, col-1]
            else:
                # Piecewise If Statements
                a1 = distance[row-1, col] + 1 
                a2 = distance[row, col-1] + 1
                a3 = distance[row-1, col-1] + 1
                distance[row, col] = min(a1,a2,a3)
    return(distance, distance[-1,-1]) 


# In[16]:


mak = 'TTAACTTCGGTCCGCACCCTAGGGCGGCTCGGG'
mak = mak.lower()
lev_string(mak, new_thing[3])[1]


# In[17]:


seq_dis = pd.DataFrame(columns = ["original", "closest", "distance"])
count = 0                       
for seq in isms['seq'][:]:
    new_temp = []
    for t in new_thing[:]:
        new_temp.append(lev_string(seq, t)[1])
    t_array = [isms['name'][count], seq, thing[0][np.argmin(new_temp)], new_temp[np.argmin(new_temp)]]
    seq_cur = pd.DataFrame([t_array], columns = ["name", "original", "closest", "distance"], index =[count])
    seq_dis = seq_dis.append(seq_cur)
    count += 1
seq_dis


# In[18]:


may_19 = 'TTAACTTCGGTCCGCACCCTAGGGCGGCTCGGG'
may_19 = may_19.lower()
lev_string(may_19, new_thing[3])[1]


# Can you replace ambiguous bases with nonambiguous ones based on other ISMs from the wastewater project?
# 
# Which region/country in the world does the cleaned ISMs found most abundant?
# 
# These samples are collected in California in different dates. What are the most abundant subtype in California at those dates according to the ISM_df_with_correction.csv file?

# In[19]:


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


# In[20]:


pip install openpyxl


# In[ ]:





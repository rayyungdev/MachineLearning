clear all; clc;

curr_string = "AATCGAAGGCT";
[k1_hits, k1_table] = kmer(curr_string, 1); %This now gives us a list of our kmers and its associated hits

%% Now we want to get CX
%{
    This should really be renamed to posterior

    Thing I should also note... Later on as I develop the different
    classes, I will need to append all of my labels and match the different
    hits (which will give me zeros if such value doesn't exist).

    Actually, let's not do this yet. Let's import the data from our
    actual datasets so that we can actually get accurate data... No reason
    to just add on zeros. 
%}

%% Pull Data and Insert into Cell
fnames = ["Limnohabitans.fasta", "T_vulcanus_rbcl.fasta", "s_thermotolerans.fasta"];
query = 'uncultured.fasta';

seq = [];
data_list = {};
for i=1:length(fnames)
    seq =[seq; fastaread(fnames(i))];
    data_list(i,:) = {fnames(i), string(seq(i).Sequence)};
end

%% Found all the 3 mers from our training data. 

for k = 1:length(data_list)
    %Let's save memory and just use that cell array.
    [hits, table] = kmer(data_list{k,2},2);
    data_list{k,2} = hits;
end
%%

%{
    Now that we have our data, we can go onto what we needd to do before.
    Append all my datasets and match our associated kmers. We should expect
    some zeros. 

    Let's create a 4 cell array, where the top row is our unique labels.
    And the following rows are our hits.

    Maybe... not.. becuase we don't have zeros. THey have the same exact
    labels, but not the same amoutn of hits... might be easier than i
    thought

    No need to do graph smoothing because we do not have any zeros...
    yet... 
    
    Also they all have the same lenght, so taking long is signifcantly
    easier!
%}
d_labels = cellstr([data_list{1,2}{:,1}]);
d_hits = [data_list{1,2}{:,2}; data_list{2,2}{:,2}; data_list{3,2}{:,2}];
d_length = length(seq(1).Sequence)-2;
d_log = log(d_hits/d_length);
d_table = array2table(d_log, 'VariableName', d_labels);

%% Now that we have trained our data... we will add our logs later. 
qf = fastaread(query);
quer_data = qf.Sequence;
[qhits, qtable] = kmer(quer_data, 2);

%{
    Luckily, we have the same amount of values in qhits, so still no graph
    smoothing required
%}
qdata = zeros(3,1);
for i = 1:length(qhits)
    qdata = qdata + qhits{i,2}*table2array(d_table(:,i));
end
%{
    This is prediction process. Since we already have a kmer function that
    calculates the amount of hits, we simply multiply the amount of hits
    per kmer in our query data by the log values in our d_table. We then
    sum all the values together. 
%}

[~, amax] = max(qdata); %Find argmax
disp(append(query, " most likely comes from ", fnames(amax)))
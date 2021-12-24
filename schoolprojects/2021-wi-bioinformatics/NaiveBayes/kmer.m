function [k_hits, k_table] = kmer(k_string, step_size)
    %if 3mer, then it's k+2, so step_size = 2; if 2_mer, then k+1, etc etc.
    
    %Assume that k_string is a string, not char array...
    kchar = char(k_string);
    
    %This will ignore other kmers, if it doesn't exist as a combination for said k_string... We will adjust
    %this later
    
    k_list = [];
    k_length = length(kchar);
    
    for i = 1:k_length-step_size
        k_list =[k_list string(kchar(i:i+step_size))];
    end
    
    k_unique = unique(k_list);
    N = numel(k_unique);
    k_hits = {};
    
    for k = 1:N
        k_hits(k,:) = {k_unique(k), sum(k_list==k_unique(k))};
    end
    
    k_table = cell2table(k_hits, 'VariableNames', {'kmer', 'hits'});
end
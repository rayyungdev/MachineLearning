%% Preprocess Data
feature_1 = table2array(hw3data(:,1));
feature_2 = table2array(hw3data(:,2));
feature_3 = table2array(hw3data(:,3));
feature_4 = table2array(hw3data(:,4));
Category = table2array(hw3data(:,5));

N = 100
%% Generating PDF's with self created function
[dis_1_a, mu_1_a, sig_1_a] = funct.sigmean_pdf(feature_1(1:50), N, [min(feature_1), max(feature_1)]);
[dis_2_a, mu_2_a, sig_2_a] = funct.sigmean_pdf(feature_2(1:50), N, [min(feature_2), max(feature_2)]);
[dis_3_a, mu_3_a, sig_3_a] = funct.sigmean_pdf(feature_3(1:50), N, [min(feature_3), max(feature_3)]);
[dis_4_a, mu_4_a, sig_4_a] = funct.sigmean_pdf(feature_4(1:50), N, [min(feature_4), max(feature_4)]);
%%
[dis_1_b, mu_1_b, sig_1_b] = funct.sigmean_pdf(feature_1(51:100), N, [min(feature_1), max(feature_1)]);
[dis_2_b, mu_2_b, sig_2_b] = funct.sigmean_pdf(feature_2(51:100), N, [min(feature_2), max(feature_2)]);
[dis_3_b, mu_3_b, sig_3_b] = funct.sigmean_pdf(feature_3(51:100), N, [min(feature_3), max(feature_3)]);
[dis_4_b, mu_4_b, sig_4_b] = funct.sigmean_pdf(feature_4(51:100), N, [min(feature_4), max(feature_4)]);

%% Plot features A vs B
figure;
subplot(4,1,1);
plot(linspace(0, max(feature_1)+1, length(dis_1_a)), dis_1_a,'Linewidth', 1.5);
hold on;
plot(linspace(0, max(feature_1)+1, length(dis_1_b)), dis_1_b, '--', 'Linewidth', 1.5);
title('Feature 1');
legend('A', 'B');

subplot(4,1,2);
plot(linspace(0, max(feature_2)+1, length(dis_2_a)), dis_2_a,'Linewidth', 1.5);
hold on;
plot(linspace(0, max(feature_2)+1, length(dis_2_b)), dis_2_b, '--', 'Linewidth', 1.5);
title('Feature 2');
legend('A', 'B');

subplot(4,1,3);
plot(linspace(0, max(feature_3)+1, length(dis_3_a)), dis_3_a,'Linewidth', 1.5);
hold on;
plot(linspace(0, max(feature_3)+1, length(dis_3_b)), dis_3_b, '--', 'Linewidth', 1.5);
title('Feature 3');
legend('A', 'B');

subplot(4,1,4);
plot(linspace(0, max(feature_4)+1, length(dis_4_a)), dis_4_a,'Linewidth', 1.5);
hold on;
plot(linspace(0, max(feature_4)+1, length(dis_4_b)), dis_4_b, '--', 'Linewidth', 1.5);
title('Feature 4');
legend('A', 'B');

%{
At this point, we have our likelihood functions for all of our individual
features, thereby classifying them between class A or B.

Things to note: All of our variances are different (else our sqrt(var)
would all be the same). Maybe check our covariance of all matrices?

We can note that it has equal priors, given the 50 A and 50 B, so we can
ignore that.

They share the same dataset, so we can ignore that as well.
%}

O = [2, 1, 1.75, 0.5; 3, 2, 2.75, 1; 4, 3,3.75, 1.5; 5, 4, 4.75, 2];
for i = 1:4
    % Cond if A;
    cond_1_a = funct.cond_pdf(O(i,1), mu_1_a, sig_1_a^2);
    cond_2_a = funct.cond_pdf(O(i,2), mu_2_a, sig_2_a^2);
    cond_3_a = funct.cond_pdf(O(i,3), mu_3_a, sig_3_a^2);
    cond_4_a = funct.cond_pdf(O(i,4), mu_4_a, sig_4_a^2);
    
    %Cond if B;
    cond_1_b = funct.cond_pdf(O(i,1), mu_1_b, sig_1_b^2);
    cond_2_b = funct.cond_pdf(O(i,2), mu_2_b, sig_2_b^2);
    cond_3_b = funct.cond_pdf(O(i,3), mu_3_b, sig_3_b^2);
    cond_4_b = funct.cond_pdf(O(i,4), mu_4_b, sig_4_b^2);
    
    pa = prod([cond_1_a, cond_2_a, cond_3_a, cond_4_a]);
    pb = prod([cond_1_b, cond_2_b, cond_3_b, cond_4_b]);
    
    if (pa > pb)
        fprintf('Object %d is in A \n', i);
    else
        fprintf('Object %d is in B \n', i);
    end
end

%{
Objects : A, B, B, B
%}

%% Problem 2
%{
no, yes, = 0, 1; single, marry, divorce = 0, 1, 2; low, high = 0, 1;
%}
ho = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0]';
ms = [0, 1, 0, 1, 2, 1, 2, 0, 1, 0]';
ai = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]';
Y = [0, 0, 0, 0, 1, 0, 0, 1, 0, 1]';

Y_table = [ho, ms, ai, Y];
X = [0, 1, 1];
%{
Looking at our table, we already know that our priors of being defaulted pB
is 0.3 and pA is 0.7;

We only care about the scenario of not homeowner, married, and high income;

p(nho | nd) = 4/7; p(nho | d) = 3/7;

p(married | nd) = 4/4; p(married | d) = 0/4;

p(high | nd) = 4/4; p(high | d) = 0/4;

Since we're using baye's, we do not want our values to be 0, so we will
subsitute the value with 0.01;
%}


pho = [4/7, 3/7];
pm = [(4 - 0.01)/4, 0.01/4];
ph = pm;

nd = pho(1) * pm(1) * ph(1) * .7
d = pho(2) * pm(2) * ph(2) * .3

%{
Just by looking at the data we can already presume that the status of X[nh,
m, h] will not be defaulted, given that our table shows that no one with
high income or has married has ever been defaulted. However, we cannot have
our values be zero in our model else it will mess up the calculations.

With all of that said, our calculated values from our baye's classifier
shows that the probability of X being defaulted is close to zero and the
probability of it being not defaulted is 40%. This leads me to believe that
our person is most likely not going to be defaulted. 
%}

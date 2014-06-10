%% Practice generating CV indices (K-folds)
% This code needs 4 parameters: 
% 1. number of trials
% 2. number of voxels
% 3. number of K (for K-folds CV)
% 4. number of row labels
% It outpus the indices for the test set

%% Warning: This will clear the work space and variables
clear;clc;

%% You can set the parameters for CV here
% Please the dimension of the data sets
ntrials = 20;
nvoxels = 10;
% Please set the number of folds 
k = 5;
% It needs to know the number of row labels
numRL = 2;

%% Create the indices matrix
ind_A = 1:k;
ind_A = repmat(ind_A', 1, ntrials/k/numRL);
indices = reshape(ind_A',ntrials/numRL, 1);
indices = repmat(indices,numRL,1);

A = 1:ntrials;
A = repmat(A', 1, nvoxels);

for i = 1:k

ind_test = indices == i;
ind_train = ~ ind_test;

% Store the indices into a matirx for future use
indices_test(:,i) = find(ind_test);  
indices_train(:,i) = find(ind_train); 

% This part can check whether the indices assignment is correct
% It could be ignored/commented
ind_test = find(ind_test);
ind_train = find(ind_train);
disp(['This is the ' num2str(i) 'th CV, the test set indices are: '])
disp(ind_test')  

end
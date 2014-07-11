%% Practice generating CV indices (K-folds)
% This code needs 3 parameters: the number of trails, the number of voxels,
% and the number of folds of the K-folds cross validation. 
% It outpus the indices for the test set
clear;clc;

%% You can set the parameters for CV here
% Please the dimension of the data sets
ntrials = 40;
nvoxels = 20;
% Please set the number of folds 
k = 5;


%% Create the indices matrix
ind_A = 1:k;
ind_A = repmat(ind_A', 1, ntrials/k);
indices = reshape(ind_A',ntrials, 1);

A = 1:ntrials;
A = repmat(A', 1, nvoxels);


for i = 1:k

ind_test = indices == i;
ind_train = ~ ind_test;

ind_test = find(ind_test);
ind_train = find(ind_train);

disp(['This is the ' num2str(i) 'th CV, the test set indices are: '])
disp(ind_test')   

ind_Xtest = A(ind_test, :);
ind_Xtrain = A(ind_train, :);

end
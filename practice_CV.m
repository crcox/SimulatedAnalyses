%% Practice generating CV indices (K-folds)
% This code needs 3 parameters: the number of trails, the number of voxels,
% and the number of folds of the K-folds cross validation. 
% It outpus the indices for the test set
clear;clc;

%% You can set the parameters for CV here
% Please the dimension of the data sets
ntrials = 200;
nvoxels = 100;
% Please set the number of folds 
k = 5;


%% Create the indices matrix
A = 1:ntrials;
X = repmat(A', 1, nvoxels);
indices = crossvalind('Kfold', ntrials, k);

for i = 1:k

test = indices == i;
train = ~ test;

test = find(test);
train = find(train);

disp(['This is the ' num2str(i) 'th CV, the test set indices are: '])
disp(test')

Xtest = X(test, :);
Xtrain = X(train, :);

end
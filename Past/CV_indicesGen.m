%% CV indices generator
% This code needs 4 parameters: 
% 1. number of trials
% 2. number of voxels
% 3. number of K (for K-folds CV)
% It outpus the indices for the test sets and training sets

%% Warning: This will clear the work space and variables
clear;clc;

%% You can set the parameters for CV here
% Please the dimension of the data sets
ntrials = 100;
nvoxels = 100;
% Please set the number of folds 
k = 5;
% Create random data set 
X =  randn(ntrials,nvoxels);   

%% Cross Validation
% generate indices for CV
CV.blocks = 1:k;
CV.blocks = CV.blocks';
CV.indices = repmat(CV.blocks,[ntrials / k,1]);

CV2.blocks = 1 : k-1;
CV2.blocks = CV2.blocks';
CV2.indices = repmat(CV2.blocks,[ntrials / k,1]);


% Creating indices for test set and training set
% for i = 1: k
%     test.indices(:,i) = CV.indices == i;
%     train.indices(:,i) = CV.indices ~= i;
% end
% 
% % You can display the indices if you'd like to see
% disp(test.indices)
% disp('=====')
% disp(train.indices)


function [ CV, CV2, test, train ] = CVindices( k, ntrials )
%UNTITLED11 Summary of this function goes here
%   Detailed explanation goes here
%% Generating indices for Outer CV
% generate indices for CV
CV.blocks = 1:k;
CV.blocks = CV.blocks';
CV.indices = repmat(CV.blocks,[ntrials / k,1]);

% Generate indices for CVglmnet
CV2.blocks = 1:k-1;
CV2.blocks = CV2.blocks';
CV2.indices = repmat(CV2.blocks,[ntrials / k,1]);

% Generate indices for CV (after the test set was taken out)
CV2.blocks = 1 : k-1;
CV2.blocks = CV2.blocks';
CV2.indices = repmat(CV2.blocks,[ntrials / k,1]);


% Creating indices for test set and training set
for i = 1: k
    test.indices(:,i) = CV.indices == i;
    train.indices(:,i) = CV.indices ~= i;
end

end


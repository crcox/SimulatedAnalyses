function [ ntrials, nvoxels, k, signal, noise, rowLabels,numsignal, testSize, X ] =...
    simData( ntrials, nvoxels, k, signal, noise, numRowLabels, numsignal )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%% You can set the parameters for CV here
% Please the dimension of the data sets 
% ntrials = 150; % it has to be divisible by K
% nvoxels = 150;
% Please set the number of folds 
% k = 5;
% It needs to know the number of row labels
% ps:Currently, this program can just run normally when there are 2 rowlabels
rowLabels.num = 2;

% Set the strength of the signal 
% signal = 1;
% numsignal = 20;
% Set the strength of the noise
 
% noise = 1.5;

% It is useful to know the size for the testing set
testSize = ntrials / k ;

% Display the parameters
disp ('Parameters: ')
disp(['number of Voxels= '  num2str(nvoxels)])
disp(['number of Trials= '  num2str(ntrials)])
disp(['K = '  num2str(k) ' (for K-folds CV)' ])
disp(['Number of row labels= ' num2str(rowLabels.num)])
disp(['Signal intensity = ' num2str(signal)])
disp(['Noise intensity = ' num2str(noise)])
disp(['Number of signals = ' num2str(numsignal)])



%% Simulate the data
% Creating the background 
X.raw = zeros(ntrials, nvoxels);

% % Add some signals 
X.raw(1:ntrials/rowLabels.num,1:numsignal / 2) = X.raw(1:ntrials/rowLabels.num,1:numsignal/2) + signal;
X.raw(ntrials/rowLabels.num + 1:end, numsignal/2 + 1 : numsignal) = X.raw(ntrials/rowLabels.num + 1 : end ,numsignal/2 + 1 : numsignal) + signal;
% plot the signal
figure(1)
imagesc(X.raw)
xlabel('Voxels');ylabel('Trials');title('Signal');

% Adding noise 
X.raw = X.raw + noise * randn(ntrials,nvoxels);   
% plot the noise + signal
figure(2)
imagesc(X.raw)
xlabel('Voxels');ylabel('Trials');title('Signal & Noise')

% Create row labels
rowLabels.whole = zeros(ntrials,1);
rowLabels.whole(1:ntrials / rowLabels.num ,1) = 1;
% imagesc(rowLabels.whole);

% Create new row labels for CV blocks
rowLabels.train = zeros(ntrials - testSize,1); 
rowLabels.train(1: (ntrials - testSize)/rowLabels.num ,1) = 1; 
rowLabels.test = zeros(testSize,1); 
rowLabels.test(1:testSize / rowLabels.num ,1) = 1; 


end


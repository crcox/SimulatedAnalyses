function [ output ] = simData( numtrials, numvoxels, numfolds, numRowLabels, signalStrength, noiseStrength )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%% You can set the parameters for CV here
% Please the dimension of the data sets 
ntrials = numtrials; % it has to be divisible by K
nvoxels = numvoxels;
% Please set the number of folds 
k = numfolds;
% It needs to know the number of row labels
% ps:Currently, this program can just run normally when there are 2 rowlabels
rowLabels.num = 2;
% It is useful to know the size for the testing set

test.size = ntrials / k; 

% Display the parameters
disp ('Parameters: ')
disp(['Voxels: '  num2str(nvoxels)])
disp(['trials: '  num2str(ntrials)])
disp(['K : '  num2str(k)])
disp(['Number of row labels: ' num2str(rowLabels.num)])


%% Simulate the data
% Creating the background 
X.raw = zeros(ntrials, nvoxels);

% % Add some signals 
signal = signalStrength;
disp(['Signal intensity = ' num2str(signal)])
X.raw(1:ntrials/rowLabels.num,1:5) = X.raw(1:ntrials/rowLabels.num,1:5) + signal;
X.raw(ntrials/rowLabels.num + 1:end,6:10) = X.raw(ntrials/rowLabels.num + 1 : end ,6:10) + signal;
% plot the signal
figure(1)
imagesc(X.raw)
xlabel('Voxels');ylabel('Trials');title('Signal');

% Adding noise 
% rng(1) % To make the result replicable
noise = noiseStrength;
disp(['Noise intensity = ' num2str(noise)])
X.raw = X.raw + noise * randn(ntrials,nvoxels);   
size(X.raw);
% plot the noise + signal
figure(2)
imagesc(X.raw)
xlabel('Voxels');ylabel('Trials');title('Signal & Noise')

% Create row labels
rowLabels.whole = zeros(ntrials,1);
rowLabels.whole(1:ntrials / rowLabels.num ,1) = 1;
% Create new row labels for CV blocks
rowLabels.train = zeros(ntrials - test.size,1); 
rowLabels.train(1: (ntrials - test.size)/rowLabels.num ,1) = 1; 
rowLabels.test = zeros(test.size,1); 
rowLabels.test(1:test.size / rowLabels.num ,1) = 1; 


end


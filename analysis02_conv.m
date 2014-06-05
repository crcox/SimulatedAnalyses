%% Convolutional T test
% Convolution method is able to consider neighbouring units when finding
% signal. This analysis will use convolution method evaluate two kinds of
% data sets.
%
% In the first data set (X1), the simluated signals sitting next to each
% other, while in the second data set (X2), the signals are far away from
% each other. It is predicted that convolution method will detect the
% signals form the fist data set more easily.
clear;clc

%% 1. Generating random data sets
ntrials = 20;
nvoxels = 40;
X = randn(ntrials,nvoxels);
X1 = X;
X2 = X;
[m,n] = size(X);


%% 2. Adding some signals
X1(1:10,1) = X1(1:10,1) + 1;   % Adding signals neighbouring to each other
X1(11:20,2) = X1(11:20,2) + 1;

X2(1:10,1) = X2(1:10,1) + 1;   % Adding signals that far away from each other
X2(11:20,11) = X2(11:20,11) + 1;


%% 3. Convolve the signal.
% Convolve each row with the weights [0.2, 0.6, 0.2]. This weights
% determine the degree of blurring.

[m,n] = size(X); % creates indices for FOR LOOP

% Convolve X1
for i = 1 : m    
    Y1(i,:) = conv(X1(i,:), [.2, .6, .2], 'same');
    
%     if i == m    % disp the result after all rows were convolved
%         disp(Y1)
%     end
end

% Convolve X2
for i = 1 : m    
    Y2(i,:) = conv(X2(i,:), [.2, .6, .2], 'same');
    
%     if i == m   % disp the result after all rows were convolved
%         disp(Y2)
%     end
end


%% 4. Conduct T test. 
% We expect h1 will be more likely to detect the signal, with the help of
% convolution.
h1 = ttest(X1(1:10,:), X1(11:20,:))
h2 = ttest(X2(1:10,:), X2(11:20,:))

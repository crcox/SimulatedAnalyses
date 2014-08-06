%% Simulate 3D data
clear; clc; close all;


% Set some parameters
length = 10;
numTrials = 100;
numVoxels = length ^ 3;

% Assign length for x y z 
% (currently it only works for space with equal length)
x = length;
y = length;
z = length;

% Get coordinates for x y z
tempx = repmat([1:x]', [y * z,1]);
tempy = repmat(1:x, z);
tempy = tempy(:);
tempz = sort(tempx);

% Concatenation
map = [tempx, tempy, tempz];
size(map);

% check if I am doing the right thing
scatter3(tempx,tempy,tempz);


% Create
X = zeros(numVoxels, numTrials);



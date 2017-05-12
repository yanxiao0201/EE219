clear all; close all; clc;
if strfind(path,'nmfv1_4')
    addpath('nmfv1_4');
end
disp('Reading data...');
R=dlmread('out.csv');
% R(R == 0) = nan;

disp('Calculating weight...');
% Weight
W=(R==0);

%W=isnan(R);
%R(W)=0;
W=~W;

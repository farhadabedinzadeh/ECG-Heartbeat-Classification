clc;clear
close
%% '================ Written by Farhad AbedinZadeh ================'
%                                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% Loading Data

imdsTrain = imageDatastore('train', ...
    'IncludeSubfolders',true, ...
    'LabelSource','none');

files = imdsTrain.Files;
parts = split(files,filesep);
labels_train=parts(:,(end-2));
imdsTrain.Labels = categorical(labels_train);


imdsTest = imageDatastore('test', ...
    'IncludeSubfolders',true, ...
    'LabelSource','none');

files = imdsTest.Files;
parts = split(files,filesep);
labels_test=parts(:,(end-2));
imdsTest.Labels = categorical(labels_test);


%% Resizing for CNN

inputSize = [227 227];
imdsTrain.ReadFcn = @(loc)imresize(imread(loc),inputSize);
% imshow(preview(imds));
%%convert gray to rgb
imdsTrain.ReadFcn = @(loc)cat(3,imread(loc),imread(loc),imread(loc));
% imshow(preview(imds));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%resizing
inputSize = [227 227];
imdsTest.ReadFcn = @(loc)imresize(imread(loc),inputSize);
% imshow(preview(imds));
%%convert gray to rgb
imdsTest.ReadFcn = @(loc)cat(3,imread(loc),imread(loc),imread(loc));
% imshow(preview(imds));

%%
num_images=length(imdsTrain.Labels);

% Visualize random 20 images
perm=randperm(num_images,25);
figure;
for idx=1:20
    f=figure;
    subplot(4,5,idx);
    imshow(imread(imdsTrain.Files{perm(idx)}));
    title(sprintf('%s',imdsTrain.Labels(perm(idx))))
end
exportgraphics(s,'subplot2.png','Resolution',1080)
%% CNN
net = alexnet;
% plot(net)
% netplot=gca;
% exportgraphics(netplot,'cnnplot.png','Resolution',2048)

%% Layers
% analyzeNetwork(net)

inputSize = net.Layers(1).InputSize;

InSizeCNN=['Input Size of this Network is: ', num2str(inputSize)];
disp(InSizeCNN)
%% Converting images to size of its input to suit the architecture
augimdsTrain = augmentedImageDatastore([227 227],imdsTrain);
augimdsValid = augmentedImageDatastore([227 227],imdsTrain);

%% Layers to transfer
layersTransfer = net.Layers(1:end-3);

% inputSize = net.Layers(1).InputSize


numClasses = numel(categories(imdsTrain.Labels));

noOfClass=['Number of classes: ', num2str(numClasses)];
disp(noOfClass)

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%%
% options = trainingOptions('sgdm', ...
%     'MiniBatchSize',10, ...
%     'MaxEpochs',6, ...
%     'InitialLearnRate',1e-4, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',augimdsValid, ...
%     'ValidationFrequency',3, ...
%     'Verbose',false, ...
%     'Plots','training-progress');

options = trainingOptions('adam','MaxEpochs',2,'MiniBatchSize',1000,...
'Plots','training-progress','Verbose',0,'ExecutionEnvironment','parallel',...
'ValidationData',augimdsValid,'ValidationFrequency',50,'ValidationPatience',3);

netTransfer = trainNetwork(augimdsTrain,layers,options);
% save netTransfer
%% Testing Phase

augimdsTest = augmentedImageDatastore([227 227],imdsTest);

% Predict Test Labels
[predicted_labels,posterior] = classify(netTransfer,augimdsTest);

% Actual Labels
actual_labels = imdsTest.Labels;

%% Confusion Matrix and ROC curve
plotconfusion(actual_labels,predicted_labels)
title('Confusion Matrix: AlexNet');
confplot=gca;
exportgraphics(confplot,'confplot.png','Resolution',2048)

%% Metrics Report
% Accuracy = sum(predicted_labels == actual_labels)/numel(actual_labels)*100;
% display(Accuracy)

% figure
% confusionchart(actual_labels,predicted_labels,'ColumnSummary','column-normalized',...
%               'RowSummary','row-normalized','Title','Confusion Chart for LSTM');

confMat = confusionmat(actual_labels, predicted_labels);
% display(confMat)

ACC = (sum(diag(confMat))/sum(sum(confMat)))*100;
SEN = (confMat(2,2)/(confMat(2,1)+confMat(2,2)))*100; 
PRE = (confMat(2,2)/(confMat(2,2)+confMat(1,2)))*100; 
F1 = (2*confMat(2,2))/(2*confMat(2,2)+confMat(2,1)+confMat(1,2))*100;

% Compute AUC
cgt = double(actual_labels);
cscores = double(posterior);
[X,Y,T,area,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(cgt,cscores(:,1),1);
AUC= area*100;

disp(['Classification Accuracy is ', num2str(ACC),'%'])
disp(['Classification Sensitivity is ', num2str(SEN),'%'])
disp(['Classification Precision is ', num2str(PRE),'%'])
disp(['Classification F1-Measure is ', num2str(F1),'%'])
disp(['Classification AUC is ', num2str(AUC),'%'])


% This script will perform the entire training sequence
% Before you run this code, make sure that you are in the right directory 

% set directory path for code and data files
clc;
clear;
directories;

% training faces list
training_faces_path = [training_directory, 'training_faces'];
training_faces_list = dir(training_faces_path);
training_faces_list = remove_directories_from_dir_list(training_faces_list);

% training nonfaces list
training_nonfaces_path = [training_directory, 'training_nonfaces'];
training_nonfaces_list = dir(training_nonfaces_path);
training_nonfaces_list = remove_directories_from_dir_list(training_nonfaces_list);

% get sizes of lists
num_faces = size(training_faces_list, 1);
num_nonfaces = size(training_nonfaces_list, 1);

%%
% First step of the training procedure is to use rectangle filters with adaboost
% on all training images

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%%  Train Using Adaboost & Rectangle filters
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
% Generate a set of cropped faces measuring 60x60 and roughly the same number
% of non-faces to use for training. By default each training image measures 100x100
% for our particular data set.

% Calculate integrals within the for loop for all training samples. 

% The integral image is used as a quick and effective way of calculating the
% sum of values (pixel values) in a given image ? or a rectangular subset of 
% a grid (the given image). It can also, or is mainly, used for calculating 
% the average intensity within a given image. If one wants to use the integral
% image, it is normally a wise idea to make sure the image is in greyscale first.

% Create cell array to store cropped faces from training set
cropFaces = cell(3047,1);
% Create cell array to store face integrals calculated from training samples
faceIntegrals = cell(1,3047);

% iterate through all number of faces to crop and generate integral images.
for i = 1:num_faces-1
    
    face2Crop = getfield(training_faces_list(i),'name');
    photo = read_gray(face2Crop);
    % crop face images with respect to the center
    centroid = (size(photo)/2)/2;
    trainingpatch = imcrop(photo, [centroid 59 59]);
    % add cropped images to cell array
    cropFaces{i} = trainingpatch;
    % calculate integral image from cropped faces
    A = cropFaces{i,1};
    B = integral_image(A);
    %figure(i); imshow(B, []);
    faceIntegrals{1, i} = B;
  
end

save cropFaceImages cropFaces;
%%

% Create cell array to store cropped non-faces from training set
cropNonFaces = cell(130,20);
% Create cell array to store non-face integrals calculated from training samples
integralNonFaces = cell(130,20);
% number of patches to crop from each non-face image
numOfPatches = 20;


for i = 1:num_nonfaces-1
    
    nonFace2Crop = getfield(training_nonfaces_list(i),'name');
    photo = read_gray(nonFace2Crop);
    
    [h w]= size(photo);
    L = 60;
    
    for j = 1:numOfPatches
        
        cropNonFaces{i,j} = photo(randi(h-L+1)+(0:L-1),randi(w-L+1)+(0:L-1));
        integralNonFaces{i,j} = integral_image(cropNonFaces{i,j});
    end
            
end

save crop_non_face_images cropNonFaces;

%%
% transform integral NonFaces cell array to 1 x 2600
NonFacesintegral = {};
for i =1:num_nonfaces-1
  for j =1:20
      
    NonFacesintegral{end+1} = integralNonFaces{i,j};  
  
  end
    
end

%%
% Initialize dimensions for face images
face_horizontal = 60;
face_vertical = 60;

% save this preprocessor data to load in test file once bootstrapping &
% cascading are applied. Uncomment and run code below to save.


%%
%generate 5000 random classifiers  
number = 5000;
weak_classifiers = cell(1,number);
for i = 1:number
    weak_classifiers{i} = generate_classifier(face_horizontal, face_vertical);
end

% save these weak classifiers to load in test file once bootstrapping &
% cascading are applied. Uncomment and run code below to save.



save classifiers5000 weak_classifiers


%%
%  precompute responses of all training examples on all weak classifiers

% store size of all training samples/patches
example_number = (num_faces-1) + (numOfPatches * (num_nonfaces-1));
% initialize vector measuring example_number x 1 with zeros
labels = zeros(example_number, 1);
% label 3047 face samples as 1
labels (1:num_faces-1) = 1;
% label 130 non-face samples as -1
labels((num_faces):example_number) = -1;
examples = zeros(face_vertical, face_horizontal, example_number);

%%
%convert cell array to matrix
faceIntegralArray = zeros(face_vertical,face_horizontal,num_faces-1);
for i = 1: num_faces-1
    faceIntegralArray(:,:,i) = cell2mat(faceIntegrals(i));
end
examples(:, :, 1:num_faces-1) = faceIntegralArray;

%convert cell array to matrix
NonfaceIntegralArray = zeros(face_vertical,face_horizontal,(numOfPatches*130));
for i = 1: 2600
    NonfaceIntegralArray(:,:,i) = cell2mat(NonFacesintegral(i));
end


save intergrals NonfaceIntegralArray faceIntegralArray
%%
examples(:, :, num_faces:example_number) = NonfaceIntegralArray;
% numel returns the number of elements, n, in array weak_classifiers
classifier_number = numel(weak_classifiers);
responses =  zeros(classifier_number, example_number);

for example = 1:example_number
    integral = examples(:, :, example);
    for feature = 1:classifier_number
        classifier = weak_classifiers {feature};
        responses(feature, example) = eval_weak_classifier(classifier, integral);
    end
    disp(example)
end


%
save training responses labels classifier_number example_number;


%%
% pass data collected on responses, labels and number of rounds to AdaBoost
tic;
boosted_classifier = AdaBoost(responses, labels, 50);
toc;
% save boosted classifier to load in test file once bootstrapping &
% cascading are applied. Uncomment and run code below to save.
%%
save boosted50 boosted_classifier

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%%  - - - - - - - - -Bootstrapping - - - - - -
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Load in values

clear;
load cropFaceImages;
load crop_non_face_images;
load classifiers5000;
load training;
load intergrals;
load boosted50;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%- - - - -CONVERTING CELL ARRAYS TO MATRICES- - - 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Converting cell array of images 130 x 20 to cell array 2600 x 1 for easy
%readabliity 
cropNonFaces2 = {};
for i =1:130
  for j =1:20
      
    cropNonFaces2{end+1,1} = cropNonFaces{i,j};  
  
  end  
end

%%
%FACES
%convert cell array to matrix 
faceArray = zeros(60,60,3047);
for i = 1: 3047
    faceArray(:,:,i) = cell2mat(cropFaces(i));
end


%%
%NON FACES
%convert cell array to matrix
NonfaceArray = zeros(60,60,(20*130));
for i = 1: 2600
    NonfaceArray(:,:,i) = cell2mat(cropNonFaces2(i));
end

%disp(NonfaceArray(:,:,2600));

% STORE ALL FACE AND NONFACE IMAGES IN ARRAY 60x60xindex
examples(:, :, 1:3047) = faceArray;
examples(:, :, 3048:5647) = NonfaceArray;
%disp(examples(:,:,5647));

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%- - - - - -Identify misclassifications- - - - -  
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



threshold = 5;

location_missed_classified = [];
location_right_classified = [];
labels_right_classified = [];
labels_of_misclassified = [];


%%
for i =1:5647
    
    photo = examples(:,:,i);
    result =apply_classifier_aux(photo, boosted_classifier, weak_classifiers, [60 60]);
                          
    class = max(max(result));

    label = labels(i,1);
    if (label == 1 && class <= threshold)
       location_missed_classified(end+1,1) = i;
       labels_of_misclassified(end+1,1) = label;
    
    elseif(label == -1 && class > threshold)
        
       location_missed_classified(end+1,1) = i;
       labels_of_misclassified(end+1,1) = label;
             
    else
        location_right_classified(end+1,1) = i;
        labels_right_classified(end+1,1) = label;
    end
end
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%- - - - -RETRAIN MISCLASSIFICATIONS- - - - - - 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
newExamples = [];
num_misclassified = size(location_missed_classified,1);
labels2 = [];

for i = 1:num_misclassified
    value = location_missed_classified(i,1);
    newExamples(:,:,i) = examples(:,:,value);
    labels2(end+1,1) = labels_of_misclassified(i,1);
    
    
end


%% add some values that were classified correctly 
for i=1:1000
   
    value = location_right_classified(i+4000,1);
    newExamples(:,:,end+1) = examples(:,:,value);
    labels2(end+1,1) = labels_right_classified(i+4000,1);
end



%%
%generate 3500 random classifiers  
number = 1000;
weak_classifiers2 = cell(1,number);
for i = 1:number
    weak_classifiers2{i} = generate_classifier(60, 60);
end

save classifiersB3500 weak_classifiers2;
%%
%classifier_number = numel(weak_classifiers2);
%responses2 =  zeros(classifier_number, 1059);

%for example = 1:1059
    %integral = newExamples(:, :, example);
    %for feature = 1:classifier_number
        %classifier2 = weak_classifiers2 {feature};
        %responses2(feature, example) = eval_weak_classifier(classifier2, integral);
    %end
    %disp(example)
%end




%%
% pass data collected on responses, labels and number of rounds to AdaBoost

%boosted_classifier2 = AdaBoost(responses2, labels2, 50);


%save BOOT_boosted_classifier boosted_classifier2


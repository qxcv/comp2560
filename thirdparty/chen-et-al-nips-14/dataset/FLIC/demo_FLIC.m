load examples.mat
imgdir = './images/';
%% display a random image
i = randi(length(examples));

img = imread([imgdir,'/',examples(i).filepath]);
cla, imagesc(img), axis image, hold on

% display torso detected by berkeley poselets
plotbox(examples(i).torsobox,'w--')

% display all the labeled joints; median of 5 annotations by mechanical turk
myplot(examples(i).coords(:,lookupPart('lsho','lelb','lwri')),'go-','linewidth',3)
myplot(examples(i).coords(:,lookupPart('rsho','relb','rwri')),'mo-','linewidth',3)
myplot(examples(i).coords(:,lookupPart('rhip','lhip')),'bo-','linewidth',3)
myplot(examples(i).coords(:,lookupPart('leye','reye','nose','leye')),'c.-','linewidth',3)
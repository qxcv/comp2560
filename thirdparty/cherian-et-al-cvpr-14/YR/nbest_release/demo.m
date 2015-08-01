compile;

% load  model
load('PARSE_final');

%%%	partIDs = 1:26;
% Two poses are overlapping if all "joint" parts are overlapping.
partIDs = [1 2 3 5 7 10 12 14 15 17 19 22 24 26]; % indice to joints/head

imlist = dir('images/*.jpg');
for i = 1:length(imlist)
    % load and display image
    im = imread(['images/' imlist(i).name]);
    clf; imagesc(im); axis image; axis off; drawnow;

    tic;
    % detect function for generating diverse hypotheses
    boxes = detect_MM(im, model, min(model.thresh,-1),partIDs); 
    dettime = toc; 
    fprintf('\nfilename : %s ... # of detections = %d, time = %.2fs\n',imlist(i).name,size(boxes,1),toc);
    %%& Suppress overlapping poses across different scales (slow)
    % boxes = boxes(nms_pose(boxes),:);
    
    [s i] = sort(boxes(:,end),'descend');
    boxes = boxes(i,:);
    % visualize detections
    colorset = {'g','g','y','m','m','m','m','y','y','y','r','r','r','r','y','c','c','c','c','y','y','y','b','b','b','b'};
    showboxes(im, boxes(1,:),colorset); % visualize the best-scoring detection
    fprintf('Press any key to continue...');
    pause;
    
end


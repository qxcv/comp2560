 function new_merged_poses = EstimatePosesInVideo(src_path, model, numposes, config)
    global numpts_along_limb num_path_parts;
    global parent keyjoints_right keyjoints_left;
    global GetDistanceWeightsFn ComputePartPathCostsFn;

    %% set some parameters!
    flow_param = config.flow_param;
    PartIDs = config.PartIDs;
    data_store_path = config.data_store_path;
    modeltype = config.modeltype;
    data_path = config.data_path;
    video_store_path = config.video_store_path;
    data_flow_path = config.data_flow_path;
    max_poses = config.MAX_POSES;
    seqlen = config.seqlen;
    Kpaths = numposes;
    cycled_nodes = config.cycled_nodes;
    GetDistanceWeightsFn = config.GetDistanceWeightsFn;
    ComputePartPathCostsFn = config.ComputePartPathCostsFn;
    numpts_along_limb = config.numpts_along_limb;
    num_path_parts = config.num_path_parts;
    parent = model.pa;
    pose_joints = config.pose_joints; % recombination tree nodes.           

    % create a temp directory for storing some intermediate files!
    [dir_name,~]=strtok(src_path(end:-1:1),'/'); boxes_loc = dir_name(end:-1:1);    
    %seq = boxes_loc; % this is the name of the sequence.
    if ~exist(data_flow_path, 'dir')        
        mkdir(data_flow_path);      
    end    
    if ~exist([data_store_path boxes_loc], 'dir')
        mkdir([data_store_path boxes_loc]);
    end

    parent = model.pa; %use the parent definition in the model.

    %% read frames and generate the required data descriptors and poses per frame.    
    imglist = dir(src_path); 
    imglist = imglist(3:end); % exclude the directories . and .. 
    numframes = length(imglist);     
    frames = [];
    for i=1:numframes,
        im = imread([src_path '/' imglist(i).name]);
        [imsize(1),imsize(2),~]=size(im);
        if isempty(frames)
            frames = zeros(imsize(1), imsize(2), 3, numframes);
        end
        frames(:,:,:,i) = im;
    end
    numfiles = numframes;
    boxes = cell(numfiles,1); allboxes = boxes;        
    img=cell(numframes,1);

    % compute optical flow per frame pair.
    fprintf('optical flow on the frames...\n');
    fopticalflow=cell(numfiles-1,1); next=1;
    for i=1:numfiles-1
        while i+next > numfiles, next  = next - 1; end % adaptive step size selection!
        try 
              load([data_flow_path strtok(imglist(i).name,'.') '-' strtok(imglist(i+next).name,'.') '.mat'], 'flow');         
        catch        
            fprintf('flow: working on file=%d/%d\n',  i, numfiles);
            [u,v] = LDOF_Wrapper(frames(:,:,:,i), frames(:,:,:,i+next));
            flow = struct('u', {u}, 'v', {v});
            save([data_flow_path strtok(imglist(i).name,'.') '-' strtok(imglist(i+next).name,'.') '.mat'], 'flow'); % cache flow... will take significant space!
        end
        optflow = flow;
        fopticalflow{i} = optflow;
    end         

    % compute the candidate poses for every frame
    fprintf('loopy pose estimation on the frames...\n');
    for i=1:numfiles-1                                                        
        img1 = frames(:,:,:,i);
        img{i} = img1; 

        try
            load([data_store_path boxes_loc '/' strtok(imglist(i).name, '.') '.mat']);                               
        catch                                 
            fprintf('loopy pose estimation on image=%d/%d\n', i, numfiles);
            img2 = frames(:,:,:,i+1); 
            stidx = i; enidx = i+1;

            % calls our extended pose model using optical flow for frame pairs.
            poseboxes = get_pose_boxes(img1, img2, boxes_loc, data_store_path, data_flow_path, imglist, stidx, enidx, stidx, 1, model, cycled_nodes,PartIDs, flow_param);
        end
        pose_range = 1:min(length(poseboxes),max_poses);       
        allboxes{i} = poseboxes;                          
        boxes{i} = poseboxes(pose_range,:);            
    end
    
    % for the last frame, we use the previous frame flow links.
    try
        img2 = frames(:,:,:,numfiles); %imread([file_path files(numfiles).name]); % img1 is the same we got from the loop above.            
        img{numfiles} = img2;
        load([data_store_path boxes_loc strtok(imglist(i).name, '.') '.mat']);
    catch            
        stidx = numfiles-1; enidx = numfiles;
        poseboxes = get_pose_boxes(img2, img1, boxes_loc, data_store_path, data_flow_path, imglist, stidx, enidx, numfiles, -1, model, cycled_nodes, PartIDs, flow_param);            
    end
    pose_range = 1:min(length(poseboxes),max_poses);          
    boxes{numfiles} = poseboxes(pose_range,:);
    
    %% 
    % now lets do the recombination procedure.
    new_merged_poses = zeros(numfiles, size(boxes{1},2));           
    try
        load([data_store_path 'new_merged_joints_' boxes_loc '.mat'], 'new_merged_poses');
    catch
        for pp=1:length(pose_joints)
            fprintf('working on pose_joints(%d)...\n', pp);           
            dist_weights = 0; %pose_joints(pp).weights;
            keyjoints_left=pose_joints(pp).keyjoints_left; keyjoints_right=pose_joints(pp).keyjoints_right;                        
            if min(keyjoints_left)==1 || min(keyjoints_right)==1 % is it a head node? if so don't use any extra kpts.
                numpts_along_limb = 1;
            else
                numpts_along_limb = 3;
            end
            
            [~, ~, paths_left, paths_right, ~] = kshortest_subpaths(boxes, fopticalflow, img, imsize, dist_weights, dist_weights, new_merged_poses); 
   
            if ~isempty(paths_right)                   
                for i=1:numfiles         
                    for s=1:length(pose_joints(pp).keyjoints_left)
                        kj = pose_joints(pp).keyjoints_left(s); kj_range = (kj-1)*4+(1:4);
                        new_merged_poses(i, kj_range) = boxes{i}(paths_left(1,i), kj_range);
                    end

                    for s=1:length(pose_joints(pp).keyjoints_right)
                        kj = pose_joints(pp).keyjoints_right(s); kj_range = (kj-1)*4+(1:4);
                        new_merged_poses(i, kj_range) = boxes{i}(paths_right(1,i), kj_range);
                    end
                end
            end                
        end 
        save([data_store_path 'new_merged_joints_' boxes_loc '.mat'], 'new_merged_poses');
    end    
end

%function dist = compute_pose_distance_matrix2(b1, b2, chist1, chist2, opticalflow, img1,img2, imsize, keyjoints, feat_type)
function dist = compute_pose_distance_matrix2(b1, b2, b3, opticalflow1, opticalflow2, img1,img2, img3, imsize, keyjoints, feat_type, parent_pos)
    global numpts_along_limb ComputePartPathCostsFn;    
    dist = ComputePartPathCostsFn(b1, b2, imsize, [], opticalflow1, img1, img2, numpts_along_limb, keyjoints, feat_type, parent_pos);
end

%% Dynamic programming algorithm using subpaths each one on each part sequence.
function [dists, paths, paths_left, paths_right, bxs] = kshortest_subpaths(boxes, opticalflow, img, imsize, weights_left, weights_right, parent_tracks)
    global keyjoints_left keyjoints_right;
    bxs{1}=boxes;
    [paths_left, paths_right, paths] = deal([]);

    %[dists, paths] = compute_sub_paths(boxes, chists, opticalflow, img, imsize, []);

    [dist_left, path_left] = compute_sub_paths(boxes, opticalflow, img, imsize, keyjoints_left, parent_tracks);
    [dist_right, path_right] = compute_sub_paths(boxes, opticalflow, img, imsize, keyjoints_right, parent_tracks);

    dists = dist_left + dist_right; paths_left(1,:) = path_left; paths_right(1,:) = path_right;
end

function [mydist, mypath] = compute_sub_paths(boxes, opticalflow, img, imsize, keyjoints, parent_tracks)
    global numpts_along_limb num_path_parts dist_weights;
    plen = length(img)-1; pps=linspace(1,plen, num_path_parts+1);    
    [mydist, mypath] = deal([]); idx=[];
   
    for r=1:num_path_parts
        ppidx = ceil(pps(r):pps(r+1)+1);
        idx = [idx, ppidx];
        C = {};      
        %dist_weights = [0 1 0 0 0 1 0.1];
        dist_weights = Get_Distance_Weights(boxes(ppidx), keyjoints);
        
        for m=2:length(ppidx)
             C{m-1} = compute_pose_distance_matrix2(boxes{ppidx(m-1)}, boxes{ppidx(m)}, [], ... 
                 opticalflow{ppidx(m-1)}, [], img{ppidx(m-1)}, img{ppidx(m)}, [], ...
                 imsize, keyjoints, dist_weights,  parent_tracks(m-1:m,:)); 
        end
                          
        [dt, pt] = ksp(C);
        mydist = [mydist, dt];
        mypath = [mypath, pt];
    end
    [~,good_idx] = unique(idx);
    mydist = mydist(good_idx); mypath = mypath(good_idx);
end

% function that computes poses using our new graphical model.
function poseboxes = get_pose_boxes(img1, img2, boxes_loc, data_store_path, data_flow_path, imglist, stidx, enidx, save_id, dirfactor, model, cycled_nodes, PartIDs, flow_param)
    load([data_flow_path strtok(imglist(stidx).name,'.') '-' strtok(imglist(enidx).name,'.') '.mat'], 'flow');     
    fim(:,:,1) = flow.u; fim(:,:,2) = flow.v;
    poseboxes = loopy_detect_MM(img1, img2, fim*dirfactor, model, -1, cycled_nodes, PartIDs, flow_param);     
    [~,idx] = sort(poseboxes(:,end), 'descend');
    idx = idx(1:min(length(idx),1000)); poseboxes = poseboxes(idx,:);
    save([data_store_path boxes_loc '/' strtok(imglist(save_id).name, '.') '.mat'], 'poseboxes');    
end


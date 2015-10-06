 function new_merged_poses = EstimatePosesInVideo(src_path, cy_model, config)
    global num_path_parts;
    global keyjoints_right keyjoints_left;
    global GetDistanceWeightsFn;

    %% set some parameters!
    data_store_path = config.data_store_path;
    data_flow_path = config.data_flow_path;
    max_poses = config.MAX_POSES;
    GetDistanceWeightsFn = config.GetDistanceWeightsFn;
    num_path_parts = config.num_path_parts;
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
    boxes = cell(numfiles,1);      
    img=cell(numframes,1);

    %% compute optical flow per frame pair.
    fprintf('optical flow on the frames...\n');
    fopticalflow=cell(numfiles-1,1);
    % These make our indices consistent in the below loop, thereby allowing
    % parfor to make fewer copies of `frames` and `imglist`.
    next_frames = frames(:,:,:,2:end);
    next_imglist = imglist(2:end);
    p = gcp; % Start parpool first, to preserve timings
    fprintf('Parallel pool up with %i workers\n', p.NumWorkers);
    fullFlowStart = tic;
    parfor i=1:numfiles-1
        dest_mat = [data_flow_path strtok(imglist(i).name,'.') ...
                '-' strtok(next_imglist(i).name,'.') '.mat'];
        try 
            loaded_flow = load(dest_mat, 'flow');
            fopticalflow{i} = loaded_flow.flow;
        catch        
            fprintf('flow: working on file=%d/%d\n',  i, numfiles);
            indivFlowStart = tic;
            [u,v] = LDOF_Wrapper(frames(:,:,:,i), next_frames(:,:,:,i));
            fopticalflow{i}.u = u;
            fopticalflow{i}.v = v;
            save_flow(dest_mat, fopticalflow{i});
            indivFlowTime = toc(indivFlowStart);
            fprintf('[T] Flow for frame %i took %fs\n', i, indivFlowTime);
        end
    end
    fullFlowTime = toc(fullFlowStart);
    fprintf('[T] All flow took %fs\n', fullFlowTime);
    
    %% run the CNN over each image for part presence
    % Cache various statistics derived from model
    [components, apps] = modelcomponents(cy_model);
    fprintf('CNN forward propagation and head position estimation on the frames...\n');
    cnnStart = tic;
    for i=1:numfiles
        box_dest_path = [data_store_path boxes_loc '/cnn_box_' strtok(imglist(i).name, '.') '.mat'];
        current_im = frames(:,:,:,i);
        img{i} = current_im;
        try
            loaded_box = load(box_dest_path);
            box = loaded_box.box;
            % Next line will throw error if we don't have enough poses
            box = box(1:max_poses, :);
        catch % there are no cached pose boxes available
            cnn_dest_path = [data_store_path boxes_loc '/cnn_pyra_' strtok(imglist(i).name, '.') '.mat'];
            
            try
                cnn_output = load(cnn_dest_path);
                pyra = cnn_output.pyra;
                unary_map = cnn_output.unary_map;
                idpr_map = cnn_output.idpr_map;
            catch % we didn't cache the CNN image pyramid
                fprintf('CNN forward propagation on image=%d/%d\n', i, numfiles);
                cnnDetStart = tic;
                % this pushes our image through the CNN at several scales
                % to make a feature pyramid for unaries and IDPR terms
                [pyra, unary_map, idpr_map] = imCNNdet(current_im, cy_model, config.gpuID, 1, @impyra);
                cnnDetStop = toc(cnnDetStart);
                fprintf('[T] imCNNdet() on %i took %fs\n', i, cnnDetStop);
                pyra = rmfield(pyra, 'feat');
                % This will take a long time and a heap of disk space, but I
                % don't care
                save(cnn_dest_path, 'pyra', 'unary_map', 'idpr_map');
            end
            
            % Now detect poses within the frame using CNN-provided unaries
            % and IDPRs
            fprintf('Intra-frame detection on image=%d/%d\n', i, numfiles);
            ifdStart = tic;
            box = in_frame_detect(...
                max_poses, pyra, unary_map, idpr_map, length(cy_model.components), ...
                components, apps, config.nms_thresh, config.nms_parts);
            ifdStop = toc(ifdStart);
            fprintf('[T] IFD on %i took %fs\n', i, ifdStop);
            save(box_dest_path, 'box');
        end
        
        boxes{i} = box;
    end
    cnnTime = toc(cnnStart);
    fprintf('[T] All detection stuff took %fs\n', cnnTime);
    
    % Make sure that we have a uniform number of pose estimates per frame
    [pose_counts, ~] = cellfun(@size, boxes);
    min_count = min(pose_counts);
    fprintf('Minimum number of pose estimations: %i\n', min_count);
    assert(min_count > 0);
    
    for i=1:length(boxes)
        % ksp.cpp FREAKS OUT if it doesn't get a double array
        % interestingly, it just hangs instead of, say, causing a page
        % fault
        boxes{i} = double(boxes{i}(1:min_count, :));
    end
    
    %% now lets do the recombination procedure.
    new_merged_poses = zeros(numfiles, size(boxes{1},2));           
    try
        load([data_store_path 'new_merged_joints_' boxes_loc '.mat'], 'new_merged_poses');
    catch
        recombStart = tic;
        for pp=1:length(pose_joints)
            fprintf('working on pose_joints(%d)...\n', pp);
            keyjoints_left=pose_joints(pp).keyjoints_left; keyjoints_right=pose_joints(pp).keyjoints_right;                        
            if min(keyjoints_left)==1 || min(keyjoints_right)==1 % is it a head node? if so don't use any extra kpts.
                numpts_along_limb = 1;
            else
                numpts_along_limb = 3;
            end
            
            [~, ~, paths_left, paths_right, ~] = kshortest_subpaths(...
                boxes, fopticalflow, img, imsize, new_merged_poses, ...
                numpts_along_limb, config.ComputePartPathCostsFn, ...
                cy_model.pa); 
   
            if ~isempty(paths_right)                   
                for i=1:numfiles         
                    for s=1:length(pose_joints(pp).keyjoints_left)
                        kj = pose_joints(pp).keyjoints_left(s);
                        kj_range = (kj-1)*4+(1:4);
                        new_merged_poses(i, kj_range) = boxes{i}(paths_left(1,i), kj_range);
                    end

                    for s=1:length(pose_joints(pp).keyjoints_right)
                        kj = pose_joints(pp).keyjoints_right(s);
                        kj_range = (kj-1)*4+(1:4);
                        new_merged_poses(i, kj_range) = boxes{i}(paths_right(1,i), kj_range);
                    end
                end
            end                
        end 
        recombTime = toc(recombStart);
        fprintf('[T] Recombination took %fs\n', recombTime);
        save([data_store_path 'new_merged_joints_' boxes_loc '.mat'], 'new_merged_poses');
    end    
 end

%%
function dist = compute_pose_distance_matrix2(b1, b2, opticalflow1, ...
    img1,img2, imsize, keyjoints, feat_type, parent_pos, ...
    numpts_along_limb, ComputePartPathCostsFn, parent)

    dist = ComputePartPathCostsFn(b1, b2, imsize, [], opticalflow1, ...
        img1, img2, numpts_along_limb, keyjoints, feat_type, ...
        parent_pos, parent);
end

%% Dynamic programming algorithm using subpaths each one on each part sequence.
function [dists, paths, paths_left, paths_right, bxs] = kshortest_subpaths(...
    boxes, opticalflow, img, imsize, parent_tracks, numpts_along_limb, ...
    ComputePartPathCostsFn, parent)

    global keyjoints_left keyjoints_right;
    bxs{1}=boxes;
    [paths_left, paths_right, paths] = deal([]);

    %[dists, paths] = compute_sub_paths(boxes, chists, opticalflow, img, imsize, []);

    [dist_left, path_left] = compute_sub_paths(boxes, opticalflow, img, ...
        imsize, keyjoints_left, parent_tracks, numpts_along_limb, ...
        ComputePartPathCostsFn, parent);
    [dist_right, path_right] = compute_sub_paths(boxes, opticalflow, img, ...
        imsize, keyjoints_right, parent_tracks, numpts_along_limb, ...
        ComputePartPathCostsFn, parent);

    dists = dist_left + dist_right; paths_left(1,:) = path_left; paths_right(1,:) = path_right;
end

function [mydist, mypath] = compute_sub_paths(boxes, opticalflow, img, ...
    imsize, keyjoints, parent_tracks, numpts_along_limb, ...
    ComputePartPathCostsFn, parent)

    global num_path_parts dist_weights;
    plen = length(img)-1; pps=linspace(1,plen, num_path_parts+1);    
    [mydist, mypath] = deal([]); idx=[];
   
    for r=1:num_path_parts
        ppidx = ceil(pps(r):pps(r+1)+1);
        idx = [idx, ppidx];
        C = cell(length(ppidx)-1, 1);
        dist_weights = Get_Distance_Weights(boxes(ppidx), keyjoints);
        
        for m=2:length(ppidx)
             C{m-1} = compute_pose_distance_matrix2(boxes{ppidx(m-1)}, boxes{ppidx(m)}, ... 
                 opticalflow{ppidx(m-1)}, img{ppidx(m-1)}, img{ppidx(m)}, ...
                 imsize, keyjoints, dist_weights,  parent_tracks(m-1:m,:), ...
                 numpts_along_limb, ComputePartPathCostsFn, parent); 
        end
                          
        [dt, pt] = ksp(C);
        mydist = [mydist, dt];
        mypath = [mypath, pt];
    end
    [~,good_idx] = unique(idx);
    mydist = mydist(good_idx); mypath = mypath(good_idx);
end

% save_flow() is trivial, but it has the advantage of being able to be
% called in a parfor (unlike save())
function save_flow(destination, flow)
save(destination, 'flow');
end
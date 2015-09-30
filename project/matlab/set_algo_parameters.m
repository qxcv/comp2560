% set some configuration settings
function config = set_algo_parameters()
%% set the configuration parameters: You need to set this.

% set the cache for storing optical flow and pose candidates. You need a
% large disk space for this.
config.cache_path = './cache/';

% this is the place to store the pose candidates
config.data_store_path = [config.cache_path 'boxes_public1/']; % 

% this is the place where the video sequences are stored as frames, each
% sequence in a separate folder. If you use poses in the wild, then the
% path will be something like below.
config.data_path = './dataset/selected_seqs/';


% cache for storing a video of the detections, this might not be
% available in the  current package.
config.video_store_path = [config.cache_path 'video/'];

% cache for flow.
config.data_flow_path = [config.cache_path 'flow/'];

% GPU ID to use for CNN evaluation. -1 to disable GPU
config.gpuID = 1;

%% Intra-frame GM parameters
% max candidate poses to use per frame
config.MAX_POSES = 150;
% poses will be ignored if any of the parts in nms_parts have detection
% boxes overlapping by more than nms_threshold with a higher scoring
% pose
config.nms_thresh = 0.95;
% part IDs to perform NMS on (currently just wrists)
config.nms_parts = [7 15];

%% Eval parameters
% Which thresholds should we use when calculating PCK statistics?
% Cherian et al. use 15:5:40, but some people use different thresholds. For
% example, Pfister et al. use 0:X:20, where X is a really small step. I think
% I'll ultimately extend the below to 0:5:40. Perhaps 0:2.5:40? The only
% challenge is getting others' results; getting Anoop's results are easy, but I
% also want to compare to Pifster et al., Yang & Ramanan, etc.
config.eval_pix_thresholds = 15:5:40;

%% If using another dataset, you might need to get the respective pose parameters 
% and set it appropriately in this function.
config.GetDistanceWeightsFn = @Get_Distance_Weights; % a function that returns 
% the weights to be used for each regularization (check practical extension
% in the paper)
config.ComputePartPathCostsFn = @Compute_SkelFlow; % a function that returns 
% the costs from practical extensions.

config.numpts_along_limb = 3; % number of extra keypoints per limb--see practical extensions
config.num_path_parts = 1; % number of sequence paths to compute per body part.

% recombination tree structure. This will change depending on the skeleton
% used in the Y&R algorithm.
config.pose_joints =  get_recombination_tree();
end

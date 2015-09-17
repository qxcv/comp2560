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

%% some internal parameters for pose estimation
config.MAX_POSES = 300; % max candidate poses to use per frame
config.flow_param = 1e-5; % weight to be used in loopy belief propagation between framepairs.
config.PartIDs = [2 3 5 7 8 10 11]; % part ids for nbest pose estimation
config.modeltype = ''; % what model are we working with. default is 13 part.
config.seqlen = 15; % some internal parameter.
config.cycled_nodes = [5,10]; % graphical nodes on which the single cycles are added.

%% If using another dataset, you might need to get the respective pose parameters 
% and set it appropriately in this function.
config.GetDistanceWeightsFn = @Get_Distance_Weights; % a function that returns 
% the weights to be used for each regularization (check practical extension
% in the paper)
config.ComputePartPathCostsFn = @Compute_SkelFlow; % a function that returns 
% the costs from practical extensions.

config.numpts_along_limb = 3; % number of extra keypoints per limb--see practical extensions
config.num_path_parts = 1; % number of sequence paths to compute per body part.

config.pix_thresh = 15; % absolute threshold for pixerror threshold

% recombination tree structure. This will change depending on the skeleton
% used in the Y&R algorithm.
config.pose_joints =  get_recombination_tree();
end


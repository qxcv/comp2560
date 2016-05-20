% set some configuration settings
function config = set_algo_parameters()
%% set the configuration parameters: You need to set this.

% set the cache for storing optical flow and pose candidates. You need a
% large disk space for this.
config.cache_path = './cache/';
%config.cache_path = '/scratch2/bigimbaz/cherian/cherian/temp/cache/IP/';

% this is the place to store the pose candidates
config.data_store_path = [config.cache_path 'boxes_public1/']; % 
config.mpii_data_store_path = [config.cache_path 'boxes_public_mpii/']; % 

% this is the place where the video sequences are stored as frames, each
% sequence in a separate folder. If you use poses in the wild, then the
% path will be something like below.
config.data_path = './dataset/selected_seqs/';
config.mpii_dest_path = './dataset/mpii/'; % where we store all MPII stuff
config.mpii_data_path = fullfile(config.mpii_dest_path, 'selected_seqs/');
%config.data_path = '/scratch2/bigimbaz/cherian/cherian/INRIA-Pose/selected_seqs/';


% cache for storing a video of the detections, this might not be
% available in the  current package.
config.video_store_path = [config.cache_path 'video/'];

% cache for flow.
config.data_flow_path = [config.cache_path 'flow/'];
config.mpii_data_flow_path = [config.cache_path 'flow_mpii/'];

% for translating MPII cooking into PIW-like structure
config.mpii_trans_spec = struct(...
    'indices', {...
        ... MIDDLE:
        12,    ... Chin (head lower point)  #1
        ... LEFT:
        4,     ... Left shoulder            #2
        [4 6], ... Left upper arm           #3
        6,     ... Left elbow               #4
        [6 8], ... Left forearm             #5
        8,     ... Left wrist               #6
        ... RIGHT:
        3,     ... Right shoulder           #7
        [3 5], ... Right upper arm          #8
        5,     ... Right elbow              #9
        [5 7], ... Right forearm            #10
        7,     ... Right wrist              #11
        ... TORSO:
        1,     ... Torso upper point        #12
        2      ... Torso lower point        #13
    }, ...
    'weights', {...
        1,         ... Chin (head lower point)  #1
        1,         ... Left shoulder            #2
        [1/2 1/2], ... Left upper arm           #3
        1,         ... Left elbow               #4
        [1/2 1/2], ... Left forearm             #5
        1,         ... Left wrist               #6
        1,         ... Right shoulder           #7
        [1/2 1/2], ... Right upper arm          #8
        1,         ... Right elbow              #9
        [1/2 1/2], ... Right forearm            #10
        1,         ... Right wrist              #11
        1,         ... Torso upper              #12
        1,         ... Torso lower              #13
    });
config.mpii_scale_factor = 0.35;


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
config.eval_pix_thresholds = 0:2.5:100;
% recombination tree structure. This will change depending on the skeleton
% used in the Y&R algorithm.
config.pose_joints =  get_recombination_tree();
end


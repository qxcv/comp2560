function hacky_watch_detections(seq)

if nargin < 1
    seq = 'seq15';
end

addpath ./visualization/;
addpath ./utils/;
config = set_algo_parameters();
load([config.data_store_path 'detected_poses_' seq], 'detected_poses');
seq_dir = [config.data_path seq '/'];    
frames = dir([seq_dir '/*.png']);
load('data/CY/CNN_Deep_13_graphical_model.mat', 'model');
parent = model.pa;
show_pose_sequence(seq_dir, frames, detected_poses, parent);
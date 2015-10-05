function hacky_watch_detections(seqs, dest_dir)

if nargin < 1
    seqs = {'seq15'};
end

addpath ./visualization/;
addpath ./utils/;
config = set_algo_parameters();
load('data/CY/CNN_Deep_13_graphical_model.mat', 'model');
parent = model.pa;

for i=1:length(seqs)
    seq = seqs{i};
    load([config.data_store_path 'detected_poses_' seq], 'detected_poses');
    seq_dir = fullfile(config.data_path, seq, '/');    
    frames = dir([seq_dir '/*.png']);
    if nargin < 2
        show_pose_sequence(seq_dir, frames, detected_poses, parent);
    else
        show_pose_sequence(seq_dir, frames, detected_poses, parent, fullfile(dest_dir, seq));
    end
end
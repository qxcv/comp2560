addpath ./visualization/
addpath ./utils/
config = set_algo_parameters();
seq = 'seq15';
load([config.data_store_path 'detected_poses_' seq], 'detected_poses');
seq_dir = [config.data_path seq '/'];    
frames = dir([seq_dir '/*.png']);
load('data/CY/CNN_Deep_13_graphical_model.mat', 'model');
parent = model.pa;
third_frame = reshape(detected_poses(3, 1:end-2), 4, []).';
first_col = third_frame(:, 1);
fprintf('Zeros:\n');
disp(find(first_col == 0).');
fprintf('Non-zeros:\n');
disp(find(first_col ~= 0).');
show_pose_sequence(seq_dir, frames, detected_poses, parent);
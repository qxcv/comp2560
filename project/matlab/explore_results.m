%% Support code for qualitative comparison
% Rank all of the sequences by PCK@15px averaged over wrists, elbows and
% shoulders to find some really terrible detections. I'd also like to find
% some frames which are individually terrible.

%% Startup code
pck_thresh = 15;
addpath ./eval/
addpath ./visualization/
addpath ./utils/
config = set_algo_parameters();
piw_data = get_piw_data('piw', config.data_path);
seqs = dir(config.data_path); seqs = seqs(3:end);
detected_pose_type = struct('seq', {}, 'filename', {}, 'frame', {}, 'bestpose',{});
detected_pose_seqs = repmat(detected_pose_type, [1,1,1]);
all_detections = []; gt_all = []; mov = 1;
seq_pck = zeros(length(seqs), 11);

%% Now go over the whole sequence
for s=1:length(seqs)
    fprintf('Attempting for sequence %s (#%i)\n', seqs(s).name, s);
    seq_dir = [config.data_path seqs(s).name '/'];
    frames = dir([seq_dir '/*.png']);
    gt = get_groundtruth_for_seq(frames, piw_data);
    gt_all = [gt_all, gt];
    load([config.data_store_path 'detected_poses_' seqs(s).name], 'detected_poses');

    for i=1:length(frames)
        detected_pose_seqs(mov, s, i).seq = seqs(s).name;
        detected_pose_seqs(mov, s, i).filename = frames(i).name;
        detected_pose_seqs(mov, s, i).frame = get_framenum(frames(i).name);
        
        if ~isempty(detected_poses)
            detected_pose_seqs(mov, s, i).bestpose = detected_poses(i,:);
        else
            detected_pose_seqs(mov, s, i).bestpose = [];
        end
    end

    seq_pck(s, :) = evaluate_pose_seqs(detected_pose_seqs(:, s:s, :), gt, pck_thresh);
end

% Average out shoulder, elbow and wrist accuracy
mean_seq_pcks = mean(seq_pck(:, [2 7 4 9 6 11]), 2);
[~, worst_idxs] = sort(mean_seq_pcks);
fprintf('Worst sequences (by mean PCK of wrists, shoulders and elbows)\n');
for i=worst_idxs'
    fprintf('Sequence %s: %f%%\n', seqs(i).name, mean_seq_pcks(i)*100);
end
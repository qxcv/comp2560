% % Copyright (C) 2014 LEAR, Inria Grenoble, France
%
% Permission is hereby granted, free of charge, to any person obtaining
% a copy of this software and associated documentation files (the
% "Software"), to deal in the Software without restriction, including
% without limitation the rights to use, copy, modify, merge, publish,
% distribute, sublicense, and/or sell copies of the Software, and to
% permit persons to whom the Software is furnished to do so, subject to
% the following conditions:
%
% The above copyright notice and this permission notice shall be
% included in all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
% NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
% LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
% OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
% WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

%%
% If you use this code, please cite:
% @INPROCEEDINGS{cherian14,
% author={Cherian, A. and Mairal, J. and Alahari, K. and Schmid, C.},
% booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
% title={Mixing Body-Part Sequences for Human Pose Estimation},
% year={2014}
% }
%
% demo code main file to run.
% for bugs contact anoop.cherian@inria.fr

function results = demo_mpii_cooking(seq_num)

startup(); % set some paths

% configure. See the function for details. You need to set the cache and
% the sequence paths in the config.
config = set_algo_parameters();
config.data_store_path = config.mpii_data_store_path;
config.data_path = config.mpii_data_path;
config.data_flow_path = config.mpii_data_flow_path;

% Load trained model from Chen & Yuille
cy_mat = load('./data/CY/CNN_Deep_13_graphical_model.mat');
cy_model = cy_mat.model;
% Set path to trained CNN appropriately
% TODO: I should move this stuff into config and pass it to imCNNdet
% directly
cy_model.cnn.cnn_conv_model_file = './data/CY/fully_conv_net_by_net_surgery.caffemodel';
cy_model.cnn.cnn_deploy_conv_file = './CY/external/my_models/flic/flic_deploy_conv.prototxt';

% get the dataset and gt annotations
mpii_test_seqs = get_mpii_cooking(...
    config.mpii_dest_path, config.cache_path, config.mpii_trans_spec);
mpii_data = translate_mpii_seqs(mpii_test_seqs, config.mpii_data_path, ...
    config.mpii_scale_factor);

% process each sequence, here we show only for seq15 available in the
% dataset folder.
detected_pose_type = struct('seq', {}, 'filename', {}, 'frame', {}, 'bestpose',{});
detected_pose_seqs = repmat(detected_pose_type, [1,1,1]);
if ~exist('seq_num', 'var')
    seqs = dir(config.data_path); seqs = seqs(3:end);
else
    % Use * so that we only get the listing for the specific directory that
    % we want. Frig, this is hacky. Oh, and you can't put the wildcard
    % *before* the last component of the filename, because Matlab.
    seqs = dir(fullfile(config.data_path, sprintf('se*q%i', seq_num))); 
    if isempty(seqs)
        error('Couldn''t find seq ''%s*''', spec_seq);
    elseif length(seqs) > 1
        error('Too many matches for seq ''%s*'' (%i)', spec_seq, length(seqs));
    end
end
all_detections = []; gt_all = []; mov = 1;

% for every sequence in the selected_seqs folder (with the dataset)
for s=1:length(seqs)
    fprintf('working on sequence %s\n', seqs(s).name);

    % read the frames and store the respective groundtruth annotations.
    seq_dir = [config.data_path seqs(s).name '/'];
    frames = dir([seq_dir '/*.png']);
    gt = get_groundtruth_for_seq(frames, mpii_data);% extract gt annotations for the frames in seq
    gt_all = [gt_all, gt]; % used for full evaluation.

    % now we are ready to compute the part sequences and recombination!
    try
        load([config.data_store_path 'detected_poses_' seqs(s).name], 'detected_poses');
    catch
        estimationStart = tic;
        detected_poses = EstimatePosesInVideo(seq_dir, cy_model, config);
        estimationTime = toc(estimationStart);
        fprintf('[T] Complete pose estimation process took %fs\n', estimationTime);
        save([config.data_store_path '/detected_poses_' seqs(s).name], 'detected_poses');
    end

    % fill in some structure for the complete evaluation
    for i=1:length(frames)
        detected_pose_seqs(mov, s, i).seq = seqs(s).name;
        detected_pose_seqs(mov, s, i).filename = frames(i).name;
        frame_match = regexp(frames(i).name, '-(?<frame_no>\d+)\.png$', 'names');
        assert(numel(frame_match) == 1);
        detected_pose_seqs(mov, s, i).frame = str2double(frame_match.frame_no);
        assert(~isnan(detected_pose_seqs(mov, s, i).frame));

        % if we could find a pose sequence path
        if ~isempty(detected_poses)
            detected_pose_seqs(mov, s, i).bestpose = detected_poses(i,:);
        else
            detected_pose_seqs(mov, s, i).bestpose = [];
        end
    end
    % show_pose_sequence(seq_dir, frames, detected_poses);
end

fprintf('flattening into other format\n');
dap = get_annotated_poses(detected_pose_seqs, gt_all);
det = piw_transback(dap);
% Results will be a cell array of cell arrays, each containing poses.
% There's one top-level cell per sequence and one bottom-level cell per
% frame.
results = test_seq_transback(det, gt_all, mpii_test_seqs, config.mpii_scale_factor);
% Now put back in original MPII format
mpii_conv_pose = @(p) [nan([2 2]); p([7 2 9 4 11 6], :); nan([4 2])];
seq_transback = @(s) cellfun(mpii_conv_pose, s, 'UniformOutput', false);
results = cellfun(seq_transback, results, 'UniformOutput', false);
!mkdir -p results/mpii
% Save all the things!
save('results/mpii/comp2560-mpii-dets', 'results', 'mpii_test_seqs', ...
    'dap', 'gt_all', 'config');
end % end functionl

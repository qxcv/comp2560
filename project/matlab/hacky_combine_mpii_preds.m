function hacky_combine_mpii_preds
%HACKY_COMBINE_MPII_PREDS Looks at per-frame CY predictions and merges into
%coherent structure
startup;
config = set_algo_parameters;
fprintf(['Warning: I assume you''ve already run demo_mpii to ' ...
    'completion.\nThis script relies on cached files from demo_mpii, ' ...
    'so it won''t work unless you have.\n']);

% get the dataset and gt annotations
mpii_test_seqs = get_mpii_cooking(...
    config.mpii_dest_path, config.cache_path, config.mpii_trans_spec);

results = cell([1 length(mpii_test_seqs.seqs)]);
mpii_conv_pose = @(p) [nan([2 2]); p([7 2 9 4 11 6], :); nan([4 2])];
unscale_pose = @(p) (p - 1) / config.mpii_scale_factor + 1;
for seq_idx=1:length(mpii_test_seqs.seqs)
    seq_path = fullfile(config.mpii_data_store_path, sprintf('seq%i', seq_idx));
    seq = mpii_test_seqs.seqs{seq_idx};
    seq_results = cell([1 length(seq)]);
    for frame_idx=1:length(seq)
        datum = mpii_test_seqs.data(seq(frame_idx));
        % See translate_mpii_seqs for details
        frame_fn = sprintf('cnn_box_mpii-pose-%04i-%09i.mat', ...
            frame_idx, datum.frame_no);
        frame_path = fullfile(seq_path, frame_fn);
        loaded = load(frame_path);
        
        % Convert boxes to PIW format (struct with .score and .point
        % fields)
        cell_boxes = mat2cell(loaded.box, ones([1 size(loaded.box, 1)]));
        piw_point_struct = piw_transback(cell_boxes);
        
        % Scale and translate each pose back into MPII format
        unscaled_poses = cellfun(unscale_pose, {piw_point_struct.point}, ...
            'UniformOutput', false);
        mpii_poses = cellfun(mpii_conv_pose, unscaled_poses, ...
            'UniformOutput', false);
        rscores = {piw_point_struct.score};
        seq_results{frame_idx} = struct('point', mpii_poses, 'score', rscores);
    end
    results{seq_idx} = seq_results;
end

!mkdir -p topk/mpii
save('topk/mpii/cy-topk-mpii-dets.mat', 'results', 'mpii_test_seqs', 'config');
fprintf('Saved to topk/mpii\n');
end

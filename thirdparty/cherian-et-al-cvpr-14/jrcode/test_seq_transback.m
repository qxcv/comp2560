function results = test_seq_transback(det, gt, test_seqs, scale_factor)
%TEST_SEQ_TRANSBACK Transform detected poses into something useful for eval

results = cell([1 length(test_seqs.seqs)]);

for seq_idx=1:length(test_seqs.seqs)
    seq = test_seqs.seqs{seq_idx};
    seq_results = cell([1 length(seq)]);
    
    for frame_idx=1:length(seq)
        frame_no = test_seqs.data(seq(frame_idx)).frame_no;
        gt_idx = find([gt.frame] == frame_no);
        assert(isscalar(gt_idx));
        pose = det(gt_idx).point;
        % We scale the images before running detections, so we have to
        % scale back to get joint coordinates in original image
        scaled_pose = (pose - 1) / scale_factor + 1;
        seq_results{frame_idx} = scaled_pose;
    end
    
    results{seq_idx} = seq_results;
end
end

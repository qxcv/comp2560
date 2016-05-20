function like_piw = translate_mpii_seqs(test_seqs, symlink_dir, scale_factor)
%TRANSLATE_MPII_SEQS Move into format expected by rest of pipeline
% Does something terrifying with copying, simply because
% EstimatePosesInVideo uses data that way :(
like_piw = [];
for seq_idx=1:length(test_seqs.seqs)
    fprintf('Sequence %i/%i\n', seq_idx, length(test_seqs.seqs));
    seq = test_seqs.seqs{seq_idx};
    seq_dest = fullfile(symlink_dir, sprintf('seq%i', seq_idx));
    if ~exist(seq_dest, 'dir')
        mkdir(seq_dest);
    end
    for frame_idx=1:length(seq)
        % Start by copying over image
        datum_idx = seq(frame_idx);
        datum = test_seqs.data(datum_idx);
        frame_fn = sprintf('mpii-pose-%04i-%09i.png', frame_idx, datum.frame_no);
        frame_dest = fullfile(seq_dest, frame_fn);
        if ~exist(frame_dest, 'file')
            % Also rescale to 560x(something), roughly, to fit in with PIW data
            orig_im = readim(datum);
            scaled_im = imresize(orig_im, scale_factor);
            imwrite(scaled_im, frame_dest);
        end
        
        % Now make new datum
        piw_datum.im = frame_dest;
        piw_datum.frame = datum.frame_no;
        piw_datum.imname = frame_fn;
        piw_datum.epi = 1;
        piw_datum.clipindex = datum_idx;
        piw_datum.point = scale_factor * (datum.joint_locs - 1) + 1;
        piw_datum.visible = true([1 size(datum.joint_locs, 1)]);
        
        like_piw = [like_piw piw_datum]; %#ok<AGROW>
        
        % Show progress for each frame with line of dots
        fprintf('.');
    end
    fprintf('\n');
end
end


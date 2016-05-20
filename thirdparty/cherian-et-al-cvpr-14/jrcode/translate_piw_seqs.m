function like_piw = translate_piw_seqs(test_seqs, copy_dir)
%TRANSLATE_PIW_SEQS Like translate_mpii_seqs for full PIW seqs
like_piw = [];
for seq_idx=1:length(test_seqs.seqs)
    fprintf('Sequence %i/%i\n', seq_idx, length(test_seqs.seqs));
    seq = test_seqs.seqs{seq_idx};
    seq_dest = fullfile(copy_dir, sprintf('seq%i', seq_idx));
    if ~exist(seq_dest, 'dir')
        mkdir(seq_dest);
    end
    for frame_idx=1:length(seq)
        % Start by copying over image
        datum_idx = seq(frame_idx);
        datum = test_seqs.data(datum_idx);
        frame_fn = sprintf('piw-pose-%04i-%09i.png', frame_idx, datum.frame_no);
        frame_dest = fullfile(seq_dest, frame_fn);
        copyfile(datum.image_path, frame_dest);
        
        % Now make new datum
        piw_datum.im = frame_dest;
        piw_datum.frame = datum.frame_no;
        piw_datum.imname = frame_fn;
        piw_datum.epi = 1;
        piw_datum.clipindex = datum_idx;
        piw_datum.point = datum.joint_locs;
        piw_datum.visible = true([1 size(datum.joint_locs, 1)]);
        
        like_piw = [like_piw piw_datum]; %#ok<AGROW>
        
        % Show progress for each frame with line of dots
        fprintf('.');
    end
    fprintf('\n');
end
end


function seqs = pairs2seqs(dataset, min_length)
% Use known pair numbers to find contiguous sequences for same scene.
% Assumes that pairs were generated with a frame skip of 1, and accounting
% for scenes (so there are no pairs which cross scene boundaries). This is
% a safe assumption for my data so far (MPII, H3.6M), but may not hold
% elsewhere.
pair_idxs = [[dataset.pairs.fst]; [dataset.pairs.snd]]';
assert(ismatrix(pair_idxs) && size(pair_idxs, 2) == 2);
sorted_idxs = sortrows(pair_idxs);
assert(all(unique(sorted_idxs(:, 1)) == sorted_idxs(:, 1)), ...
    'Cannot have duplicate pair starts');
% Following should succeed because the frame skip is uniform
assert(all(unique(sorted_idxs(:, 2)) == sorted_idxs(:, 2)), ...
    'Cannot have duplicate pair ends');
assert(all(sorted_idxs(:, 1) < sorted_idxs(:, 2)), ...
    'First frame must have lower ID than second');

% Indexes from frame number to sequence number
seq_map = containers.Map('KeyType', 'double', 'ValueType', 'double');
seq_num = 1;
seqs = {};
for i=1:size(sorted_idxs, 1)
    start = sorted_idxs(i, 1);
    finish = sorted_idxs(i, 2);
    if seq_map.isKey(start)
        this_seq = seq_map(start);
        seq_map(finish) = seq_map(start);
    else
        this_seq = seq_num;
        seq_map(finish) = this_seq;
        seqs{this_seq} = start; %#ok<AGROW>
        seq_num = seq_num + 1;
    end
    seqs{this_seq} = [seqs{this_seq} finish]; %#ok<AGROW>
end

% Trim out sequences that are too small, assert that everything is sorted
% and sane.
is_sane = @(arr) ~isempty(arr) && all(arr == unique(arr));
assert(all(cellfun(is_sane, seqs)));
% Turns out that to do logical indexing into a cell array, you need to use
% parentheses. Huh.
seqs = seqs(cellfun(@length, seqs) >= min_length);

% Where sequences overlap, always choose the largest one
starts = cellfun(@min, seqs);
ends = cellfun(@max, seqs);
lengths = cellfun(@length, seqs);
should_keep = true([1 length(seqs)]);

for seq_num=1:length(starts)
    start = starts(seq_num);
    finish = ends(seq_num);
    this_length = lengths(seq_num);
    collides = (start <= ends) & (finish >= starts);
    collides(seq_num) = false;
    
    % Give priority to sequences which (a) are longer or (b) are equal
    % length and come before this one
    if any(lengths(collides) > this_length) ...
            || any(find((lengths == this_length) & collides) < seq_num)
        should_keep(seq_num) = false;
    end
end

seqs = seqs(should_keep);
end

function rv = make_test_set(some_dataset, test_seqs)
% Turn ordinary dataset with some defined sequences into a test set (which
% has sequences but not pairs).
rv = some_dataset;
rv.seqs = test_seqs;
rv = rmfield(rv, {'pairs', 'num_pairs'});
rv.name = [rv.name '_seqs_for_test'];
end

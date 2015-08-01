function labels = derive_labels(note, clusters, imdata, tsize)
% derive pairwise relational type labels
conf = global_conf();
pa = conf.pa;

cachedir = conf.cachedir;

label_name = sprintf('%s_labels_K%d.mat', note, conf.K);

try
  load([cachedir label_name]);
catch
  % assign mix
  labels = assign_label(imdata, clusters, pa, tsize, conf.K);
  save([cachedir label_name], 'labels', '-v7.3');
end

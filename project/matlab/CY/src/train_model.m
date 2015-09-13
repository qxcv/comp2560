function model = train_model(note, pos_val, neg_val, tsize)
conf = global_conf();
cachedir = conf.cachedir;
pa = conf.pa;
cls = [note '_graphical_model'];
try
  model = parload([cachedir cls], 'model');
catch
  % learn clusters, and derive labels
  % must have already been learnt!
  clusters = learn_clusters();
  label_val = derive_labels('val', clusters, pos_val, tsize);
  % ------------
  for ii = 1:numel(label_val)
    if isfield(label_val(ii), 'invalid')
      assert(all(~label_val(ii).invalid));    % currently, all validation labels should be valid !!!
    end
  end
  
  model = build_model(pa, clusters, tsize);
  % add near filed to provide mixture supervision
  for ii = 1:numel(pos_val)
    pos_val(ii).near = label_val(ii).near;
  end
  caffe('reset');
  model = train(cls, model, pos_val, neg_val, 1);
  caffe('reset');
  parsave([cachedir cls], model);
end

function [flat_dets, flat_gts] = flatten_to_cells(dets, gts)
%FLATTEN_TO_CELLS Turn crazy structs into simple cells of joint coords
% Useful for using my new evaluation code (github:qxcv/joint-regressor).
flat_gts = {gts.point};
detected_annotated_poses = get_annotated_poses(dets, gts);
gt_format_det = piw_transback(detected_annotated_poses);
flat_dets = {gt_format_det.point};
end


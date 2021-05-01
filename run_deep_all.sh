for segmentation_type in no, corase, fine; do
  for drop_user_features in False, True; do
    for split_by_expert in False, True; do
      python run_deep.py grid_search $segmentation_type $split_by_expert $drop_user_features
    done
  done
done

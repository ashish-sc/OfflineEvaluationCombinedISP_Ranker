# OfflineEvaluationCombinedISP_Ranker
This code calculates Masknet ranker score for sample data at `gs://tpu-cg-us/ranker_isp_golden_nu_Hindi/`.

This data is prepared by joining the ranker table data `maximal-furnace-783.sharechat_ranker.ranker_hitesh_real_time_embd_base_dataset_Punjabi_video` with the logged ISP table `maximal-furnace-783.sc_analytics.cand_score_diversity_log2_v2`.

Both are joined on basis of userId,postId and time in hour.

Finally this table is joined with vplay, engagement table on userId,postId and time in hour.

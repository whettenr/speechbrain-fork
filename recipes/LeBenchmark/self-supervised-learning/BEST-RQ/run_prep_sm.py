from lebenchmark_prepare import prepare_lebenchmark

base_dir = '/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/'

tr_splits = [
    # small 1k
    base_dir + 'mls_french/train',
]

dev_splits = [
    base_dir + 'mls_french/dev',
]

te_splits = [
    base_dir + 'mls_french/test',
]

merge_lst = tr_splits
merge_name = "train.csv"
save_folder = '/gpfswork/rech/nkp/uaj64gk/growth/prep_lebench/csvs/sm'
prepare_lebenchmark(
    save_folder, 
    tr_splits, 
    dev_splits, 
    te_splits, 
    merge_lst, 
    merge_name
)
from lebenchmark_prepare import prepare_lebenchmark


tr_splits = [
    # small 1k
    '/corpus/LeBenchmark/mls_french_flowbert/gpfswork/rech/zfg/commun/data/temp/mls_french/train',
    # # medium-clean – 2.7k
    # '/corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves', 
    # # medium – 3k
    # '/corpus/LeBenchmark/slr57_flowbert/African_Accented_French/wavs',
    # '/corpus/LeBenchmark/slr88_flowbert/Att-HACK_SLR88/wavs',
    # '/corpus/LeBenchmark/cafe_flowbert/CaFE/wavs',
    # '/corpus/LeBenchmark/CFPP_corrected/CFPP_corrected/output',
    # '/corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns',
    # '/corpus/LeBenchmark/gemep_flowbert/GEMEP/wavs',
    # '/corpus/LeBenchmark/mpf_flowbert/MPF/output_waves',
    # '/corpus/LeBenchmark/port_media_flowbert/PMDOM2FR_00/PMDOM2FR_wavs',
    # '/corpus/LeBenchmark/TCOF_corrected/TCOF_corrected/output',
    # # large - 7k
    # '/corpus/LeBenchmark/MaSS/MaSS/output_waves',
    # '/corpus/LeBenchmark/nccfr_flowbert/NCCFr/output_waves',
    # '/corpus/LeBenchmark/voxpopuli_unlabelled/wav',
    # '/corpus/LeBenchmark/voxpopuli_transcribed_data/wav',
    # # extra-large - 14k
    # '/corpus/LeBenchmark/audiocite_with_metadata/wavs',
    # '/corpus/LeBenchmark/studios-tamani-kalangou-french/v1_10102021/output_wav',
]

dev_splits = [
    '/corpus/LeBenchmark/mls_french_flowbert/gpfswork/rech/zfg/commun/data/temp/mls_french/dev',
]

te_splits = [
    '/corpus/LeBenchmark/mls_french_flowbert/gpfswork/rech/zfg/commun/data/temp/mls_french/test',
]


merge_lst = tr_splits
merge_name = "train.csv"
save_folder = '/users/rwhetten/attention_alt/grow-brq/lebenchmark_csvs/sm'
prepare_lebenchmark(
    save_folder, 
    tr_splits, 
    dev_splits, 
    te_splits, 
    merge_lst, 
    merge_name
)
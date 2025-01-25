from lebenchmark_prepare import prepare_lebenchmark

base_dir = '/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark'

tr_splits = [
    # small 1k
    base_dir + '/mls_french/train',
    # medium-clean – 2.7k
    base_dir + '/EPAC_flowbert/output_waves', 
    # medium – 3k
    base_dir + '/African_Accented_French/wavs',
    base_dir + '/Att-HACK_SLR88/wavs',
    base_dir + '/CaFE/wavs',
    base_dir + '/CFPP_corrected/CFPP_corrected/output',
    base_dir + '/ESLO/wav_turns',
    base_dir + '/GEMEP/wavs',
    base_dir + '/mpf_flowbert/MPF/output_waves',
    base_dir + '/port_media_flowbert/PMDOM2FR_00/PMDOM2FR_wavs',
    base_dir + '/TCOF_corrected/TCOF_corrected/output',
    # large - 7k
    base_dir + 'Mass/output_waves',
    base_dir + 'NCCFr/output_waves',
    base_dir + 'voxpopuli_unlabelled/wav',
    base_dir + 'voxpopuli_transcribed_data/wav',
    # extra-large - 14k
    base_dir + '/audiocite_with_metadata/wavs',
    base_dir + '/Niger-mali-audio-collection/output_wav',
]

dev_splits = [
    base_dir + '/mls_french/dev',
]

te_splits = [
    base_dir + '/mls_french/test',
]

merge_lst = tr_splits
merge_name = "train.csv"
save_folder = '/users/rwhetten/attention_alt/grow-brq/lebenchmark_prep_test'
prepare_lebenchmark(
    save_folder, 
    tr_splits, 
    dev_splits, 
    te_splits, 
    merge_lst, 
    merge_name
)
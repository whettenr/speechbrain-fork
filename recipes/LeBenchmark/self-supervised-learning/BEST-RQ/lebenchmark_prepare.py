"""
Data preparation for Libri-light

Author
------
 * Ryan Whetten, 2025


Download: https://github.com/facebookresearch/libri-light/tree/main/data_preparation
Directory Structure
Libri-light/
├── duplicate
├── large
├── LibriSpeech (only need dev and test sets)
├── medium
└── small
"""

import csv
import functools
import os
import random
from collections import Counter
from dataclasses import dataclass
import pandas as pd

from speechbrain.dataio.dataio import (
    merge_csvs,
    read_audio_info,
)
from speechbrain.utils.data_utils import download_file, get_all_files
from speechbrain.utils.logger import get_logger
from speechbrain.utils.parallel import parallel_map

logger = get_logger(__name__)
# OPT_FILE = "opt_librilight_prepare.pkl"
# SAMPLERATE = 16000


def prepare_lebenchmark(
    data_folders,
    save_folder,
    tr_splits=[],
    tr_save_names=[],
    dev_splits=[],
    te_splits=[],
    merge_lst=[],
    merge_name=None,
    split_interval=30,
    skip_prep=False,
):
    """
    This class prepares the csv files for the LeBenchmark datasets
    Paper: https://arxiv.org/pdf/2309.05472

    Arguments
    ---------
    data_folders : list
        Paths to the folders where the datasets are stored.
    save_folder : str
        The directory where to store the csv files.
    tr_splits : list
        List of train splits to prepare from ['small', 'medium-clean', 
        'medium', 'large', 'extra-large'].
    dev_splits : list
        List of dev splits to prepare from LibriSpeech ['dev-clean','dev-others'].
    te_splits : list
        List of test splits to prepare from LibriSpeech ['test-clean','test-others'].
    merge_lst : list
        List of splits (e.g, small, medium,..) to
        merge in a single csv file.
    merge_name: str
        Name of the merged csv file.
    split_interval: int
        Inveral in seconds to split audio files. If 0 this will result 
        in not spliting the files at all. Use 0 only if files have been
        already split (for exaple by VAD).
    skip_prep: bool
        If True, data preparation is skipped.
    
    Returns
    -------
    None

    Example
    -------
    >>> from lebenchmark_prepare import prepare_lebenchmark
    >>> data_folders = ['/corpus/LeBenchmark/mls_french_flowbert/gpfswork/rech/zfg/commun/data/temp/mls_french']
    >>> tr_splits = ['train', 'output_waves']
    >>> dev_splits = ['LibriSpeech/dev-clean']
    >>> te_splits = ['LibriSpeech/test-clean']
    >>> save_folder = '/users/rwhetten/attention_alt/grow-brq/lebenchmark_prep_test'
    >>> prepare_lebenchmark(data_folder, save_folder, tr_splits, dev_splits, te_splits)

data_folders = ['/corpus/LeBenchmark/mls_french_flowbert/gpfswork/rech/zfg/commun/data/temp/mls_french', '/corpus/LeBenchmark/epac_flowbert/EPAC_flowbert']
tr_splits = ['train', 'output_waves']
tr_save_names = ['mls', 'epac']

from lebenchmark_prepare import prepare_lebenchmark
data_folders = ['/corpus/LeBenchmark/mls_french_flowbert/gpfswork/rech/zfg/commun/data/temp/mls_french']
tr_splits = ['train']
tr_save_names = ['mls', 'epac']
dev_splits = ['LibriSpeech/dev-clean']
te_splits = ['LibriSpeech/test-clean']
merge_lst = tr_save_names
merge_name = "train.csv"
save_folder = '/users/rwhetten/attention_alt/grow-brq/lebenchmark_prep_test'
prepare_lebenchmark(data_folders, save_folder, tr_splits, tr_save_names, dev_splits, te_splits, merge_lst, merge_name)
    """

    if skip_prep:
        return
    splits = tr_splits 
    split_save_names = tr_save_names 
    # + dev_splits + te_splits
    save_folder = save_folder

    # Other variables
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Check if this phase is already done (if so, skip it)
    if skip(split_save_names, save_folder):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    # Additional checks to make sure the data folder contains Librispeech
    check_folders(data_folders, splits)
    

    # create csv for each dataset
    for data_folder, split, split_save_name in zip(data_folders, splits, split_save_names):
        wav_lst = get_all_files(
            os.path.join(data_folder, split), match_or=[".flac", ".wav"]
        )

        create_csv(save_folder, wav_lst[:20], split, split_save_name, split_interval)

        

    # Merging csv file if needed
    print("merge_lst")
    print(merge_lst)
    print("merge_name")
    print(merge_name)
    if merge_lst and merge_name is not None:
        merge_files = [spl + ".csv" for spl in merge_lst]
        print("merge_files")
        print(merge_files)
        merge_csvs(
            data_folder=save_folder, csv_lst=merge_files, merged_csv=merge_name
        )

    # logger.info("Data info...")
    # for split in splits:
    #     path = os.path.join(save_folder, os.path.basename(split) + ".csv")
    #     df = pd.read_csv(path)
    #     hours = df.duration.sum() / 3600
    #     logger.info(f'Split {split} contains {hours} hours')

    # path = os.path.join(save_folder, "train.csv")
    # df = pd.read_csv(path)
    # hours = df.duration.sum() / 3600
    # logger.info(f'Total hours in training: {hours}')


@dataclass
class LSRow:
    ID: str
    file_path: str
    start: float
    stop: float
    duration: float

def process_and_split_line(wav_file, split_interval) -> list:
    info = read_audio_info(wav_file)
    duration = info.num_frames
    split_interval = split_interval * info.sample_rate
    new_rows = []
    start = 0 
    components = wav_file.split(os.sep)
    id_name = os.path.join(components[-2], components[-1])
    print(f'looking at: {wav_file}, {duration}')
    if split_interval != 0:
        while start < duration:
            stop = min(start + split_interval, duration)
            new_rows.append([
                id_name + str(start / info.sample_rate),
                wav_file,
                start,
                stop,
                (stop - start) / info.sample_rate,
            ])
            start = start + split_interval
    else:
        new_rows.append([
            id_name,
            wav_file,
            0,
            0,
            duration / info.sample_rate,
        ])
    
    return new_rows

def process_line(wav_file) -> LSRow:
    info = read_audio_info(wav_file)
    duration = info.num_frames / info.sample_rate

    return LSRow(
        ID=os.path.basename(wav_file),
        file_path=wav_file,
        start=0,
        stop=0,
        duration=duration,
    )

def create_csv(save_folder, wav_lst, split, split_save_name, split_interval):
    """
    Create the dataset csv file given a list of wav files.

    Arguments
    ---------
    save_folder : str
        Location of the folder for storing the csv.
    wav_lst : list
        The list of wav files of a given data split.
    split : str
        The name of the current data split.
    split_save_name : str
        The name to save the csv file.
    split_interval : int
        Max len of audio.

    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, os.path.basename(split_save_name) + ".csv")
    if os.path.exists(csv_file):
        logger.info("Csv file %s already exists, not recreating." % csv_file)
        print("Csv file %s already exists, not recreating." % csv_file)
        return

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "wav", "start", "stop", "duration"]]

    # Processing all the wav files in wav_lst
    # FLAC metadata reading is already fast, so we set a high chunk size
    # to limit main thread CPU bottlenecks
    if 'dev' in split or 'test' in split:
        print('dev or test')
        logger.info(f'Processing {split}')
        for row in parallel_map(process_line, wav_lst, chunk_size=8192):
            csv_line = [
                row.ID,
                row.file_path,
                str(row.start),
                str(row.stop),
                str(row.duration),
            ]

            # Appending current file to the csv_lines list
            csv_lines.append(csv_line)
    else:
        print('in else')
        csv_lines = [["ID", "wav", "start", "stop", "duration"]]
        logger.info(f'Processing {split} and splitting into {split_interval} sec chunks...')
        print(f'Processing {split} and splitting into {split_interval} sec chunks...')
        line_processor = functools.partial(process_and_split_line, split_interval=split_interval)
        for rows in parallel_map(line_processor, wav_lst, chunk_size=128):
            # Appending current file to the csv_lines list
            csv_lines = csv_lines + rows


    # Writing the csv_lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final print
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)


def skip(splits_names, save_folder):
    """
    Detect when the Libri-light data prep can be skipped.

    Arguments
    ---------
    splits : list
        A list of the splits expected in the preparation.
    save_folder : str
        The location of the save directory

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking csv files
    skip = True

    for split in splits_names:
        if not os.path.isfile(os.path.join(save_folder, os.path.basename(split) + ".csv")):
            skip = False


def check_folders(data_folders, splits):
    """
    Check if the data folder actually contains the dataset.

    If it does not, an error is raised.

    Arguments
    ---------
    data_folder : str
        The path to the directory with the data.
    splits : list
        The portions of the data to check.

    Raises
    ------
    OSError
        If folder is not found at the specified path.
    """
    # Checking if all the splits exist
    for data_folder, split in zip(data_folders, splits):
        logger.info(f'Checking {data_folder}/{split}')
        print(f'Checking {data_folder}/{split}')
        split_folder = os.path.join(data_folder, split)
        if not os.path.exists(split_folder):
            err_msg = (
                "the folder %s does not exist (it is expected in the "
                "dataset)" % split_folder
            )
            raise OSError(err_msg)

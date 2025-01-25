"""
Data preparation for Libri-light

Author
------
 * Ryan Whetten, 2025
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
    save_folder,
    tr_splits=[],
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
        List of train splits to prepare.
    dev_splits : list
        List of dev splits to prepare from.
    te_splits : list
        List of test splits to prepare from.
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
    """

    # TODO adjust script to get savename and tr_splits from folder 
    # TODO dev set

    if skip_prep:
        return

    # splits = all train and validation datasets
    splits = tr_splits + dev_splits + te_splits
    save_folder = save_folder

    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    # Additional checks to make sure the data folder contains the dataset
    check_folders(splits)

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder):
        logger.info(f'Skipping preparation, completed in previous run.')
        return
    else:
        logger.info("Data_preparation...")
    
    # create csv for each dataset
    for split in splits:
        wav_lst = get_all_files(split, match_or=[".wav", ".flac"])
        create_csv(save_folder, wav_lst, split, split_interval)        

    # Merging csv file if needed
    if merge_lst and merge_name is not None:
        merge_files = []
        for split in splits:
            split_name = split.split('/')[-2]
            if 'dev' in split:
                split_name += '-dev'
            if 'test' in split:
                split_name += '-test'
            merge_files.append(split_name + '.csv')
        
        print("merge_files")
        print(merge_files)
        merge_csvs(
            data_folder=save_folder, csv_lst=merge_files, merged_csv=merge_name
        )

    logger.info("Data info...")
    print("Data info...")
    for split in merge_files:
        path = os.path.join(save_folder, split)
        df = pd.read_csv(path)
        hours = df.duration.sum() / 3600
        logger.info(f'Split {split} contains {hours} hours')
        print(f'Split {split} contains {hours} hours')

    path = os.path.join(save_folder, "train.csv")
    df = pd.read_csv(path)
    hours = df.duration.sum() / 3600
    logger.info(f'Total hours in training: {hours}')
    print(f'Total hours in training: {hours}')


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

def create_csv(save_folder, wav_lst, split, split_interval):
    """
    Create the dataset csv file given a list of wav files.

    Arguments
    save_folder : str
        Location of the folder for storing the csv.
    wav_lst : list
        The list of wav files of a given data split.
    split : str
        The name of the current data split.
    split_interval : int
        Max len of audio.

    Returns
    -------
    None
    """
    # Setting path for the csv file
    split_name = split.split('/')[-2]
    if 'dev' in split:
        split_name += '-dev'
    if 'test' in split:
        split_name += '-test'
    csv_file = os.path.join(save_folder, split_name + ".csv")
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


def skip(splits, save_folder):
    """
    Detect when the data prep can be skipped.

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

    for split in splits:
        split_name = split.split('/')[-2]
        if 'dev' in split:
            split_name += '-dev'
        if 'test' in split:
            split_name += '-test'
        print(f'Checking {os.path.join(save_folder, split_name + ".csv")}')
        if not os.path.isfile(os.path.join(save_folder, split_name + ".csv")):
            skip = False
    return skip


def check_folders(splits):
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
    for split in splits:
        split_name = split.split('/')[-2]
        split_audio_location = split.split('/')[-1]
        logger.info(f'Checking {split_name}/{split_audio_location}')
        print(f'Checking {split_name}/{split_audio_location}')

        if not os.path.exists(split):
            err_msg = (
                "the folder %s does not exist (it is expected in the "
                "dataset)" % split
            )
            raise OSError(err_msg)

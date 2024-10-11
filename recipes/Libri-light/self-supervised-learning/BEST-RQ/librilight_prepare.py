"""
Data preparation for Libri-light

Author
------
 * Ryan Whetten, 2024


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


def prepare_librilight(
    data_folder,
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
    This class prepares the csv files for the Libri-light dataset
    with LibriSpeech dev and test sets.
    Download link: https://github.com/facebookresearch/libri-light/tree/main/data_preparation
    Download link: http://www.openslr.org/12

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Libri-light dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    tr_splits : list
        List of train splits to prepare from ['small', 'medium', 
        'large','duplicate'].
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
    >>> data_folder = 'datasets/Libri-light'
    >>> tr_splits = ['small']
    >>> dev_splits = ['LibriSpeech/dev-clean']
    >>> te_splits = ['LibriSpeech/test-clean']
    >>> save_folder = 'librilight_prepared_test'
    >>> prepare_librilight(data_folder, save_folder, tr_splits, dev_splits, te_splits)
    """

    if skip_prep:
        return
    data_folder = data_folder
    splits = tr_splits + dev_splits + te_splits
    save_folder = save_folder

    # Other variables
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    # Additional checks to make sure the data folder contains Librispeech
    check_folders(data_folder, splits)

    # create csv files for each split
    for split_index in range(len(splits)):
        split = splits[split_index]

        wav_lst = get_all_files(
            os.path.join(data_folder, split), match_and=[".flac"]
        )

        create_csv(save_folder, wav_lst, split, split_interval)

    # Merging csv file if needed
    if merge_lst and merge_name is not None:
        merge_files = [split_libri + ".csv" for split_libri in merge_lst]
        merge_csvs(
            data_folder=save_folder, csv_lst=merge_files, merged_csv=merge_name
        )

    logger.info("Data info...")
    for split in splits:
        path = os.path.join(save_folder, os.path.basename(split) + ".csv")
        df = pd.read_csv(path)
        hours = df.duration.sum() / 3600
        logger.info(f'Split {split} contains {hours} hours')

    path = os.path.join(save_folder, "train.csv")
    df = pd.read_csv(path)
    hours = df.duration.sum() / 3600
    logger.info(f'Total hours in training: {hours}')


@dataclass
class LSRow:
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
    if split_interval != 0:
        while start < duration:
            stop = min(start + split_interval, duration)
            new_rows.append([
                wav_file,
                start,
                stop,
                (stop - start) / info.sample_rate,
            ])
            start = start + split_interval
    else:
        new_rows.append([
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
        file_path=wav_file,
        start=0,
        stop=0,
        duration=duration,
    )

def create_csv(save_folder, wav_lst, split, split_interval):
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

    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, os.path.basename(split) + ".csv")
    if os.path.exists(csv_file):
        logger.info("Csv file %s already exists, not recreating." % csv_file)
        return

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [["wav", "duration"]]

    # Processing all the wav files in wav_lst
    # FLAC metadata reading is already fast, so we set a high chunk size
    # to limit main thread CPU bottlenecks
    if 'dev' in split or 'test' in split:
        logger.info(f'Processing {split}')
        for row in parallel_map(process_line, wav_lst, chunk_size=8192):
            csv_line = [
                row.file_path,
                str(row.start),
                str(row.stop),
                str(row.duration),
            ]

            # Appending current file to the csv_lines list
            csv_lines.append(csv_line)
    else:
        csv_lines = [["wav", "start", "stop", "duration"]]
        logger.info(f'Processing {split} and splitting into {split_interval} sec chunks...')
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

    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, os.path.basename(split) + ".csv")):
            skip = False


def check_folders(data_folder, splits):
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
        logger.info(f'Checking {split}')
        split_folder = os.path.join(data_folder, split)
        if not os.path.exists(split_folder):
            err_msg = (
                "the folder %s does not exist (it is expected in the "
                "dataset)" % split_folder
            )
            raise OSError(err_msg)


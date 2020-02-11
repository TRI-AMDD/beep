# Copyright 2019 Toyota Research Institute. All rights reserved.
"""
Script to batch process all csvs used in early prediction manuscript
and generate structure jsons (processed cycler objects) and featurize them
for model training and validation

Data was stored locally, so this file will fail
"""

import json
import os
from docopt import docopt

from beep.structure_data import RawCyclerRun, ProcessedCyclerRun, add_suffix_to_filename, \
    process_file_list_from_json, EISpectrum
from beep import generate_features

from glob import glob

def batch_structure_and_featurize(file_directory,features_label='full_model', to_structure=0, to_featurize=0):
    # This can be updated if we want to read files directly from S3

    processed_json_path = os.path.join(file_directory, "processed_jsons")
    if not os.path.exists(processed_json_path):
        os.makedirs(processed_json_path)

    featurized_jsons_path = os.path.join(file_directory, "featurized_jsons")
    if not os.path.exists(featurized_jsons_path):
        os.makedirs(featurized_jsons_path)

    discards = {"batches": ["2017-05-12", "2017-06-30", "2018-04-12"],
                "channels": [["CH8_", "CH10", "CH12", "CH13", "CH22"],
                             ["CH7_", "CH8_", "CH9_", "CH15", "CH16"],
                             ["CH2_", "CH23", "CH32", "CH37", "CH38", "CH39"]]}

    if (to_structure):
        all_csvs = glob(os.path.join(file_directory, "*.csv"))
        metadata_csvs = glob(os.path.join(file_directory, "*Metadata.csv"))
        cycler_csvs = list(set(all_csvs) - set(metadata_csvs))

        for batch in  discards['batches']:
            batch_of_files = [s for s in cycler_csvs if batch in s]
            json_obj = {"file_list": batch_of_files,
                        "validity": [True] * len(batch_of_files)}
            json_string = json.dumps(json_obj)
            json_output = process_file_list_from_json(json_string, processed_dir=processed_json_path)

    if (to_featurize):
        processed_jsons = glob(os.path.join(processed_json_path, "*structure.json"))
        for batch in discards['batches']:
            batch_of_files = [s for s in processed_jsons if batch in s]
            json_obj = {"file_list": batch_of_files}
            json_string = json.dumps(json_obj)
            json_output = generate_features.process_file_list_from_json(json_string, features_label=features_label,\
                                                                        predict_only=False,\
                                                                        processed_dir=featurized_jsons_path)
            """
            KeyError  encountered for "2017-05-12_3_6C-80per_3_6C_CH4_structure.json"
            KeyError  encountered for "2017-06-30_CH14_structure.json"
            KeyError  encountered for "2018-04-12_batch8_CH26_structure.json"
            
            These cells do not have more than 1 cycle of information.

            """
def main():
    """
    Main function of this module, takes in arguments of an input
    path containing all csv files and

    Returns:
        None

    """
    # Parse args and construct initial cycler run
    args = docopt(__doc__)
    file_directory = args['INPUT_JSON']
    batch_structure_and_featurize(file_directory)
    return None


if __name__ == "__main__":
    main()



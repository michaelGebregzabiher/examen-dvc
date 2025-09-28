import requests
import os
import logging
from check_structure import check_existing_file, check_existing_folder


def import_raw_data(raw_data_relative_path, 
                    filenames,
                    bucket_folder_url):
    '''Import filenames from bucket_folder_url into raw_data_relative_path'''
    # Create directory if it doesn't exist (fixed logic)
    if not check_existing_folder(raw_data_relative_path):
        os.makedirs(raw_data_relative_path, exist_ok=True)
    
    # Download all the files
    for filename in filenames:
        # Remove accidental space from URL (critical fix)
        input_file = os.path.join(bucket_folder_url.strip(), filename)
        output_file = os.path.join(raw_data_relative_path, filename)
        
        if not check_existing_file(output_file):
            print(f'Downloading {input_file} as {os.path.basename(output_file)}')
            
            try:
                response = requests.get(input_file)
                response.raise_for_status()  # Raise exception for bad status codes
                
                # For CSV files, write as text with UTF-8 encoding
                with open(output_file, "w", encoding='utf-8') as text_file:
                    text_file.write(response.text)
                print(f"Successfully downloaded {filename}")
                
            except requests.exceptions.RequestException as e:
                print(f'Error downloading {input_file}: {e}')
        else:
            print(f'File {output_file} already exists, skipping download')


def main(raw_data_relative_path="./data/raw_data", 
        filenames=["raw.csv"],
        # Removed space before URL (critical fix)
        bucket_folder_url="https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/"          
        ):
    """Upload data from AWS s3 in ./data/raw_data"""
    import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)
    logger = logging.getLogger(__name__)
    logger.info('Raw dataset created successfully')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()

import boto3
import os
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()  # looks for a file named `.env` in the current directory

def upload_folder_to_s3(root_folder, bucket_name, s3_prefix):
    """
    Uploads all files from a local folder to an S3 bucket while preserving folder structure.

    Parameters:
        root_folder (str): Local directory path to upload from.
        bucket_name (str): Name of the S3 bucket.
        s3_prefix (str): S3 prefix (virtual folder path, e.g. 'transcripts/').
    """
    # Initialize the S3 client using environment credentials
    s3 = boto3.client('s3')
    excluded_extensions = ['.ipynb']

    for root, _, files in os.walk(root_folder):
        for file in files:
            if any(file.endswith(ext) for ext in excluded_extensions):
                continue

            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, root_folder)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")

            try:
                s3.upload_file(local_path, bucket_name, s3_key)
                print(f"✅ Uploaded: {s3_key}")
            except Exception as e:
                print(f"❌ Failed to upload {local_path}: {e}")




def combine_transcripts_and_upload(transcripts_root, bucket_name, output_s3_key):
    """
    Combines all JSON transcripts from a specified S3 prefix into a single text file
    and uploads it back to S3.
    Parameters:
        transcripts_root (str): S3 prefix where JSON transcripts are stored.
        bucket_name (str): Name of the S3 bucket.
        output_s3_key (str): S3 key for the combined output file.
    """
    # Initialize S3 client
    s3 = boto3.client('s3')

    # Local temp file to store combined output
    output_file = 'all_transcripts_combined.txt'

    # Set up pagination to get all objects under the given prefix
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=transcripts_root)

    all_texts = []

    # Loop through each JSON file and extract text
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('.json'):
                    try:
                        response = s3.get_object(Bucket=bucket_name, Key=key)
                        content = response['Body'].read().decode('utf-8')
                        data = json.loads(content)
                        segments = data.get('segments', [])
                        for segment in segments:
                            text = segment.get('text', '').strip()
                            if text:
                                all_texts.append(text)
                    except Exception as e:
                        print(f"Failed to process {key}: {e}")

    # Save all text to a local file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_texts))
    
    print(f"✅ All texts saved locally to {output_file}")

    # Upload to S3
    s3.upload_file(output_file, bucket_name, output_s3_key)
    print(f"✅ Uploaded to s3://{bucket_name}/{output_s3_key}")

    # Optional: remove local file
    os.remove(output_file)




# === Example Usage ===
if __name__ == "__main__":
    upload_folder_to_s3(
        root_folder='D:/matrix/data',
        bucket_name='salesdata1225',
        s3_prefix='transcripts/'
    )
    
    combine_transcripts_and_upload(
        transcripts_root='transcripts/Masked Transcripts(212)/',
        bucket_name='salesdata1225',
        output_s3_key='combined_transcript.txt'
        
    )


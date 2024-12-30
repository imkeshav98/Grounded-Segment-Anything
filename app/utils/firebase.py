# File: app/utils/firebase.py

import os
import firebase_admin
from firebase_admin import credentials, storage
from pathlib import Path
from dotenv import load_dotenv

class FirebaseUploader:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get the root directory path
        root_dir = Path(__file__).parents[2]  # Go up two levels from utils/firebase.py
        cred_path = root_dir / "serviceAccountKey.json"
        
        # Initialize Firebase with credentials
        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred, {
            'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET')
        })
        self.bucket = storage.bucket()

    def upload_image(self, file_content: bytes, folder: str, filename: str) -> str:
        """Simple function to upload image and return URL"""
        try:
            # Create unique filename
            ext = filename.split('.')[-1]
            file_path = f"AllFiles/GenApi/{folder}/{filename}"
            
            # Upload to Firebase Storage
            blob = self.bucket.blob(file_path)
            blob.upload_from_string(
                file_content,
                content_type=f'image/{ext}'
            )
            
            # Make the blob publicly accessible
            blob.make_public()
            
            # Return public URL
            return blob.public_url
            
        except Exception as e:
            raise Exception(f"Failed to upload image: {str(e)}")

# Initialize uploader
firebase = FirebaseUploader()
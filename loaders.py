from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Tuple, Optional
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from ast import literal_eval
import json

def image_to_df():
    # Code adapted from GeminiAI (Google 2025)
    import pandas as pd
    file_path = 'images/MIDOGpp.json'

    # 1. Load the entire JSON file into a Python dictionary
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 2. Flatten the data using 'annotations' as the list of records
    # The 'data' object here is the dictionary representing 'root'
    df = pd.json_normalize(
        data, 
        record_path='annotations'
    )
    
    # 3. Select only the columns you are interested in
    columns_to_keep = ['bbox', 'labels', 'category_id', 'image_id']
    df_final = df[columns_to_keep]
    
    # print(df_final.head())
    # print(f"\nInitial DataFrame successfully loaded with {len(df_final)} rows.")

    import pandas as pd

    file_path = 'datasets_xvalidation.csv'
    
    df = pd.read_csv(file_path, sep = ';')
    df = df.drop(columns=['Dataset'])
    # print(df.head())
    # print(f"\nExtra info DataFrame successfully loaded with {len(df_final)} rows.")
    # Merge the two dataframes to include the scanner, tumor, origin, and species in each label

    df_merged = pd.merge(
        left=df_final,
        right=df,
        left_on='image_id',     
        right_on='Slide',   
        how='inner'            
    )
    df_merged = df_merged.drop(columns=['Slide'])

    incorrect_spelling = 'Hamammatsu XR'
    correct_spelling = 'Hamamatsu XR'
    
    # Use .str.replace() to find all instances of the incorrect spelling 
    # and replace them with the correct spelling.
    df_merged['Scanner'] = df_merged['Scanner'].str.replace(
        incorrect_spelling, 
        correct_spelling
    )
    return df_merged

def create_loaders(df_merged, filters=None):

    def apply_dataframe_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        """
        Applies a dictionary of key-value filters (column: value) to a DataFrame.
        Returns the filtered DataFrame.
        """
        if not filters:
            return df # No filters to apply
    
        print(f"Applying filters: {filters}")
        
        # Start with a mask of all True values
        mask = pd.Series(True, index=df.index) 
        
        # Iterate through the dictionary and build the complex mask
        for column, value in filters.items():
            if column in df.columns:
                # Combine the current condition with the existing mask using logical AND (&)
                mask &= (df[column] == value)
            else:
                print(f"Warning: Filter column '{column}' not found in DataFrame. Skipping.")
    
        # Apply the final mask and reset the index for clean splitting later
        df_filtered = df[mask].reset_index(drop=True)
        print(f"Original size: {len(df)}. Filtered size: {len(df_filtered)}.")
        
        return df_filtered
    
    # 1. APPLY FILTERING FIRST (If filters are provided)
    if filters:
        df_merged = apply_dataframe_filters(df_merged, filters)
    
    # Check if any data remains after filtering
    if len(df_merged) == 0:
        print("ERROR: No data remaining after filtering. Cannot create loaders.")
        return None, None, None
        
    # 2. THEN DEFINE THE SPLITS (70% Train, 15% Val, 15% Test)
    # This ensures the splits are based *only* on the data you want to use.
    df_train, df_val, df_test = create_stratified_data_splits(
        df_merged, 
        test_size=0.15, 
        val_size=0.15, 
        random_state=42
    )
    
    # 3. Instantiate Datasets (No need to pass 'filters' anymore)
    IMAGE_DIR = 'cropped_images/'
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Instantiate Datasets
    train_dataset = CustomSingleAnnotationDataset(df_train, IMAGE_DIR, transform=data_transform)
    val_dataset = CustomSingleAnnotationDataset(df_val, IMAGE_DIR, transform=data_transform)
    test_dataset = CustomSingleAnnotationDataset(df_test, IMAGE_DIR, transform=data_transform)
    
    # Instantiate DataLoaders
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # No shuffle for validation
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # No shuffle for testing
    
    print("\nTrain/Validation/Test DataLoaders created successfully.")
    print(f"Train Loader batch size: {BATCH_SIZE}")
    print(f"Train Loader output shape: [Batch_Size, 3, 50, 50] (Verification)")
    
    # Optional: Verify the first batch's shape and labels
    for i, batch in enumerate(train_loader):
        print(f"\nVerification Batch 1:")
        print("  Image Batch Shape:", batch['image'].shape) 
        print("  Labels in Batch (first 5):", batch['label'][:5])
        break

    return train_loader, val_loader, test_loader

class CustomSingleAnnotationDataset(Dataset):
    """
    Loads the pre-cropped 50x50 patch using the 'patch_id' column.
    """
    def __init__(self, cleaned_df: pd.DataFrame, image_dir: str = 'cropped_images/', transform=None): # <-- Changed default dir
        
        self.data_frame = cleaned_df
        self.patch_dir = image_dir # Now pointing to 'cropped_images/'
        self.transform = transform
        
        # NOTE: We no longer need 'file_id' here, as we use 'patch_id' (DataFrame index)
        # We rely on the 'patch_id' column created during preprocessing (df.index)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        
        # The 'patch_id' column must exist in the DataFrame passed to the loader!
        patch_id = row['patch_id'] 
        
        # --- Image Loading ---
        # The filename is the patch_id (e.g., "12345.png")
        patch_name = os.path.join(self.patch_dir, f"{patch_id}.png")
        
        try:
            # 1. Load the small, pre-cropped patch directly
            cropped_image = Image.open(patch_name).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Cropped patch not found at path: {patch_name}. Run preprocessing first!")

        # 2. Data Retrieval and Label Shifting
        
        # Original label is 1 or 2. We shift it to 0 or 1.
        # This resolves the IndexError from CrossEntropyLoss.
        shifted_label = row['category_id'] - 1 
        label = torch.tensor(shifted_label, dtype=torch.long)
        
        # 3. Apply the transformation (Tensor conversion)
        if self.transform:
            cropped_image = self.transform(cropped_image) 

        sample = {
            'image': cropped_image, 
            'label': label,
        }
        
        return sample

def create_stratified_data_splits(df: pd.DataFrame, 
                                 test_size: float = 0.15, 
                                 val_size: float = 0.15, 
                                 random_state: int = 42,
                                 target_column: str = 'category_id'
                                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the cleaned annotation DataFrame into stratified Train, Validation, and Test sets.
    """
    
    # Check for stratification feasibility
    if target_column not in df.columns or df[target_column].nunique() < 2:
        print(f"Warning: Cannot perform stratified split. Using simple random split.")
        stratify_data = None
    else:
        stratify_data = df[target_column]

    # --- Step 1: Split data into Training/Validation pool and Test set ---
    # The 'stratify' argument ensures class distribution is maintained.
    df_train_val, df_test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=stratify_data
    )

    # --- Step 2: Split Training/Validation pool into Training and Validation sets ---
    # Calculate the proportion of the train_val_pool that will become the validation set.
    val_proportion_of_pool = val_size / (1.0 - test_size)
    
    df_train, df_val = train_test_split(
        df_train_val, 
        test_size=val_proportion_of_pool, 
        random_state=random_state, 
        # Stratify based on the target labels within the pool
        stratify=df_train_val[target_column]
    )

    print("\n--- Data Split Summary ---")
    print(f"Original Total Annotations: {len(df)}")
    print(f"Train Annotations: {len(df_train)} ({len(df_train) / len(df) * 100:.1f}%)")
    print(f"Validation Annotations: {len(df_val)} ({len(df_val) / len(df) * 100:.1f}%)")
    print(f"Test Annotations: {len(df_test)} ({len(df_test) / len(df) * 100:.1f}%)")
    
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


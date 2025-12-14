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


import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms # Needed for default transform
from typing import Tuple, Optional
import os

# NOTE: Assumes apply_dataframe_filters, create_stratified_data_splits, 
# and CustomSingleAnnotationDataset are defined in the environment.

def create_loaders(df_merged: pd.DataFrame, 
                   patch_dir: str = "", 
                   filters: Optional[dict] = None, 
                   train_transform=None, # Renamed for clarity
                   eval_transform=None,  # New: Deterministic transform for Val/Test
                   final_train: bool = False) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Creates and returns Train, Validation, and Test DataLoaders.

    Args:
        df_merged (...): The master DataFrame.
        patch_dir (...): Path to images.
        filters (...): DataFrame filters.
        train_transform (callable, optional): PyTorch transform pipeline for the TRAINING set.
        eval_transform (callable, optional): PyTorch transform pipeline for the VALIDATION/TEST sets (deterministic).
        final_train (bool): If True, combines Train and Validation data.
    
    Returns:
        Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]: (train_loader, val_loader, test_loader)
    """

    # ... (apply_dataframe_filters function definition remains here) ...
    # Placeholder for the helper function (needed for the context to run):
    def apply_dataframe_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        if not filters: return df
        mask = pd.Series(True, index=df.index)
        for column, value in filters.items():
             if column in df.columns: mask &= (df[column] == value)
        df_filtered = df[mask].reset_index(drop=True)
        return df_filtered
    
    # 1. APPLY FILTERING FIRST
    if filters:
        df_merged = apply_dataframe_filters(df_merged, filters)
    
    if len(df_merged) == 0:
        print("ERROR: No data remaining after filtering. Cannot create loaders.")
        return None, None, None

    # 2. DEFINE THE SPLITS (70% Train, 15% Val, 15% Test)
    # NOTE: Assumes create_stratified_data_splits is available
    df_train_full, df_val, df_test = create_stratified_data_splits(
        df_merged, 
        test_size=0.15, 
        val_size=0.15, 
        random_state=42
    )

    # 3. COMBINATION LOGIC for Final Training
    if final_train:
        df_train_source = pd.concat([df_train_full, df_val], ignore_index=True)
        df_val_source = pd.DataFrame() # Empty
        df_test_source = df_test
        val_loader = None
        
        # In final_train mode, use the provided train_transform for the large training set.
        # Use the deterministic eval_transform for the final test set report.
        train_data_transform = train_transform
        eval_data_transform = eval_transform or transforms.Compose([transforms.ToTensor()])
        
        print(f"\n--- Final Training Mode Activated (Train + Val Combined) ---")
        
    else:
        # Standard Training/Hparam Tuning Mode
        df_train_source = df_train_full
        df_val_source = df_val
        df_test_source = df_test
        val_loader = None 
        
        # Use provided train_transform for training (with augmentation)
        train_data_transform = train_transform
        # Use provided eval_transform for evaluation (deterministic)
        eval_data_transform = eval_transform

    # 4. Handle Default Transformations
    if train_data_transform is None:
        train_data_transform = transforms.Compose([transforms.ToTensor()])
        print("Warning: No train_transform provided. Defaulting to ToTensor().")

    if eval_data_transform is None:
        # If no eval_transform is given, just use the train_transform (may include augmentation)
        # This is okay since the user controls the input, but should be noted.
        eval_data_transform = train_data_transform
        print("Warning: No eval_transform provided. Validation/Test will use train_transform.")


    # 5. Define Image Directory and Batch Size
    IMAGE_DIR = patch_dir + 'cropped_images/'
    BATCH_SIZE = 32
    
    # 6. Instantiate Datasets and DataLoaders
    
    # --- Train Loader ---
    train_dataset = CustomSingleAnnotationDataset(df_train_source, IMAGE_DIR, transform=train_data_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # --- Validation Loader (Only created if not in final_train mode) ---
    if not df_val_source.empty:
        val_dataset = CustomSingleAnnotationDataset(df_val_source, IMAGE_DIR, transform=eval_data_transform) # <-- FIXED!
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # --- Test Loader ---
    if not df_test_source.empty:
        test_dataset = CustomSingleAnnotationDataset(df_test_source, IMAGE_DIR, transform=eval_data_transform) # <-- FIXED!
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    else:
        test_loader = None

    print("\nDataLoaders created successfully.")
    print(f"Train Loader batch size: {BATCH_SIZE}")
            
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


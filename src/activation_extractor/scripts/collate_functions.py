import numpy as np
import os, sys

from PIL import Image

import torch

# Collate Functions ----------------------------------------------------------------
def collate_pil(batch):
    """
    Convert PIL images to numpy arrays.
    """
    batch = [np.array(img) for img in batch]

def create_collate_mscoco(image_source,
                          modality="image-text",
                          image_dir=None,
                         ):

    def collate_mscoco(batch):
        batch_size = len(batch)
        
        text_batch = []
        image_batch = []
        
        for idx in range(batch_size):
            
            # Open the image file
            if modality in ["image-text", "image"]:
                url = batch[idx]['URL']
                
                #online
                if image_source=="online":
                    image = Image.open(requests.get(url, stream=True).raw)
                
                #locally
                elif image_source=="local":
                    image_path = url.split("/")[-1]
                    image_path = os.path.join(image_dir,image_path)
                    image = Image.open(image_path)
                
                img_array = np.array(image)
                # img_array = torch.from_numpy(img_array)
                image_batch.append(img_array)
        
            #Get text text
            if modality in ["image-text", "text"]:
                text = batch[idx]['TEXT']
                text_batch.append(text)
            
        #structure inputs
        if modality=="image-text":
            input_data = {
                        "text":text_batch,
                         "image":image_batch,
                         }
            batch = input_data  
            
        elif modality=="text":
            batch = text_batch
            
        elif modality=="image":
            #only batch size 1 for now ❗
            batch = image_batch[0]
            
        return batch
            
    return collate_mscoco

def create_collate_to_max_length(max_length):
    """
    Creates a custom collate_fn that truncates sequences to a max length.
    """
    def collate_to_max_length(batch):
        batch=[sequence[0:min(max_length, len(sequence))] for sequence in batch]
        return batch
    return collate_to_max_length

def collate_to_upper(batch):
    """
    Convert sequence to ALL CAPS.
    """
    batch = [sequence.upper() for sequence in batch]
    return batch
    
def compose_collate_fns(*args):
    """
    Compose collate functions one after the other.
    """
    def composed_collate(batch):
        for function in args:
            batch = function(batch)
        return batch
    return composed_collate
    
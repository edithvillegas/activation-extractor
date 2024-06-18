import numpy as np
import os, shutil
from collections import OrderedDict

import torch

from activation_extractor.extractors.intermediateExtractorBase import IntermediateExtractorBase
from activation_extractor.model_functions.embedding_to_numpy import embedding_to_numpy 

class IntermediateExtractor(IntermediateExtractorBase):
  """
  Extends the functionality of the ``IntermediateExtractorBase`` to automatically save activations.
  """
  
    #Save outputs ---------------------------------------------------------------------------
    def gpu_to_cpu(self):
        """
        Takes all the intermediate activations stored in the extractor object and moves them to CPU
        (if on GPU) after formatting them to a numpy array.
        """
        for name in self.layer_list:
            self.intermediate_outputs[name] = embeddings_to_numpy(self.intermediate_outputs[name])  

    def save_outputs(self, output_folder, reset=False, move_to_cpu=True, 
                     saving_type="numpy_compressed", emb_format='full'):
        """
        Save intermediate activation dictionary to output folder.
        You can choose:
        
        * the saving function (numpy_compressed or numpy) 
        * the embedding format (full, mean, LT: last token).
        """
      
        #check the format
        if emb_format not in ['full','mean','LT']:
            raise ValueError(f"format must be 'full','mean' or 'LT'; got {emb_format} instead.")

        #move from gpu to cpu
        if move_to_cpu: self.gpu_to_cpu()
            
        #make output folder
        if reset: shutil.rmtree(output_folder, ignore_errors=True)
        os.makedirs(output_folder, exist_ok=True)
     
        #save each layer 
        for name,outputs in self.intermediate_outputs.items():
            # full, mean or last token embeddings 
            match emb_format:
                case 'mean':
                    outputs = np.mean(outputs,axis=1)
                case 'LT':
                    outputs = outputs[:,-1,:]

            #different saving functions
            match saving_type:
                case "numpy_compressed":
                    np.savez_compressed(f'{output_folder}/{name}.npz',
                                outputs)
                case "numpy":
                    np.save(f'{output_folder}/{name}.npy',
                                outputs)

import numpy as np
import os, shutil
from collections import OrderedDict

import torch
from transformers.modeling_outputs import (MaskedLMOutput, 
                                            BaseModelOutputWithNoAttention, 
                                            BaseModelOutputWithPastAndCrossAttentions)

from activation_extractor.extractors.intermediateExtractorBase import IntermediateExtractorBase

# Class ====================================================================================
class IntermediateExtractor(IntermediateExtractorBase):
    #Save outputs ---------------------------------------------------------------------------
    def gpu_to_cpu(self):
        for name in self.layer_list:
            self.intermediate_outputs[name] = embeddings_to_numpy(self.intermediate_outputs[name])  

    def save_outputs(self, output_folder, reset=False, move_to_cpu=True, saving_type="numpy_compressed", emb_format='full'):
        '''
        Save intermediate activation dictionary to output folder.
        You can choose:
        - the saving function (numpy_compressed or numpy) 
        - the embedding format (full, mean, LT: last token).
        '''
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

#Functions ==============================================================================
def embedding_to_numpy(embeddings):
    """
    Converts different types of module outputs to a numpy array.
    Handles different cases for the different models.
    Additionally, moves from GPU to CPU. 

    :param embedding: Intermediate output object from a pytorch model layer/module.
    
    :return: intermediate output as a numpy array
    :rtype: numpy array
    """
    
    #convert list to array (for protT5 tokens)
    if isinstance(embeddings, list): 
        embeddings = np.array(embeddings) 
        return embeddings
    
    #convert output object to tensor -------------------------
    #intermediate activations
    if isinstance(embeddings, tuple): 
        embeddings = embeddings[0] 
        
    #NT/ESM output object
    if isinstance(embeddings, MaskedLMOutput): 
        embeddings = embeddings['logits'] 
        
    #HYENA or ANKH output object
    if (isinstance(embeddings, BaseModelOutputWithNoAttention) 
        or isinstance(embeddings, BaseModelOutputWithPastAndCrossAttentions)):
        embeddings = embeddings['last_hidden_state']  
    #convert output object to tensor -------------------------
    
    #float conversions for tensors 
    if torch.is_tensor(embeddings):
        if embeddings.dtype==torch.bfloat16: embeddings = embeddings.float() 
   
    #to CPU numpy array
    embeddings = embeddings.cpu().detach().numpy()   
    
    return embeddings
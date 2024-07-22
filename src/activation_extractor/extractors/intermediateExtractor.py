import numpy as np
import os, shutil, sys
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
            try:
                self.intermediate_outputs[name] = embedding_to_numpy(self.intermediate_outputs[name])  
            except:
                #print(f"{name} not in dictionary", file=sys.stderr)
                pass

    def emb_reformatting(self, outputs, 
                                 emb_format='full', 
                                 sequence_axis=1,
                                 custom_position=None,
                                ):
        """
        Takes an output to save and reformats it according to ``emb_format`` :
        * full: nothing
        * mean: mean along sequence axis
        * LT: last token in sequence
        * FT: first token in sequence
        * custom: custom token position in sequence
        """
        #check the format
        if emb_format not in ['full','mean','LT', 'FT','custom']:
            raise ValueError(f"format must be 'full','mean' or 'LT', 'FT', 'custom'; got {emb_format} instead.")

        token_position_dict = {
            "FT":0,
            "LT":-1,
            "custom":custom_position,
            #"full":None, "mean":None,
        }

        match emb_format:
            case 'mean':
                outputs = np.mean(outputs, axis=sequence_axis)
            
            case 'LT' | 'FT' | 'custom':
                token_position = token_position_dict[emb_format]
                
                #get slicer for the right axis ===============
                #Create a list of slice(None) for each dimension
                slicer = [slice(None)] * outputs.ndim  
                #select last token or First token on the sequence axis index 
                slicer[sequence_axis] = token_position
                #format slicer
                slicer = tuple(slicer)
                
                #slice array selecting
                outputs = outputs[slicer]

            case 'full':
                pass
                
        return outputs


    def save_outputs(self, output_folder, output_id, reset=False, move_to_cpu=True, 
       save_method="numpy_compressed", emb_formats=['LT', 'FT'],
        sequence_axis=1, custom_position=None):
        """
        Save intermediate activation dictionary to output folder.
        You can choose:
        
        * the saving function (numpy_compressed or numpy) 
        * the embedding format (full, mean, LT: last token) : a list.
        * sequence_axis : sequence length axis to take mean or last token from
        """
        
        #make output folder
        if reset: shutil.rmtree(output_folder, ignore_errors=True)
        os.makedirs(output_folder, exist_ok=True)
        os.chmod(output_folder, mode=0o777)
        for emb_format in emb_formats:
            os.makedirs(output_folder+f"/{emb_format}/{output_id}", exist_ok=True)
            os.chmod(output_folder+f"/{emb_format}/{output_id}", mode=0o777)
        
        #move from gpu to cpu
        if move_to_cpu: self.gpu_to_cpu()
        
        #save each layer 
        for name,outputs in self.intermediate_outputs.items():
            #reformat outputs
            for emb_format in emb_formats:
                #reformat outputs
                outputs = self.emb_reformatting(outputs=outputs, 
                                     emb_format=emb_format, 
                                     sequence_axis=sequence_axis,
                                     custom_position=custom_position,
                                    )
                
                #different saving functions
                match save_method:
                    case "numpy_compressed":
                        np.savez_compressed(f'{output_folder}/{emb_format}/{output_id}/{name}.npz',
                                  outputs)
                    case "numpy":
                        np.save(f'{output_folder}/{emb_format}/{output_id}/{name}.npy',
                                  outputs)

import numpy as np
import torch

from activation_extractor.extractors.intermediateExtractor import IntermediateExtractor
from activation_extractor.inferencers.inferencerBase import InferencerBase as Inferencer

class Buffer:
    cached_embs = []
    
    def __init__(self, inferencer, extractor, layer_name, sequence_list,
                 tokens_batch_size, #how many tokens to retrieve at a time
                 num_elements=100, #how many elements to infer at a time
                 batch_size=1, #only works for batch_size 1 for now   
                 sequence_axis=1, #which dim is the sequence length
                 num_iterations=1, #how many times to iterate over the dataset
                ):
        
        self.layer_name = layer_name
        self.sequence_list = sequence_list.copy() * num_iterations #dataset

        #model, extractor
        self.inferencer = inferencer
        self.extractor = extractor
        
        #configuration
        self.num_elements = num_elements #number of sequences to infer
        self.batch_size = batch_size
        self.sequence_axis =sequence_axis
        self.tokens_batch_size = tokens_batch_size

        #iterator
        self.total_length = sum( list( map(len, sequence_list) ) ) #sum of each sequence length
        self.end = int(self.total_length/self.tokens_batch_size)*num_iterations #number of possible iterations
        self.current = 0

    def fill(self, num_elements, batch_size):
        #for element in list, do inference on elements, pop element from list, append to buffer storage
        
        for element_i in range(0, num_elements):
            element = self.sequence_list[:batch_size]
            
            #inference
            processed = self.inferencer.process( element ) #tokenize
            outputs = self.inferencer.inference(processed) #inference

            #format outputs
            self.extractor.gpu_to_cpu() #format outputs
            
            #convert array to list and append
            for layer in [self.layer_name]:
                #convert matrix into list of 1D numpy arrays
                self.extractor.intermediate_outputs[layer]= [ row for row in self.extractor.intermediate_outputs[layer][0,:,:] ] 
                
            self.cached_embs += self.extractor.intermediate_outputs[self.layer_name]  #add numpy vectors to buffer

            #delete elements from dataset
            del self.sequence_list[:batch_size]

    def get_embs(self, num_tokens):
        #get a certain number of tokens from buffer, check if there are enough first, if not fill buffer
        
        check=True
        while check:
            if len(self.cached_embs)>num_tokens : #enough tokens?
                #return tokens, pop from list
                embs = self.cached_embs[:num_tokens]
                del self.cached_embs[:num_tokens]

                #break while
                check=False 
                # return embs
                return torch.tensor( np.array(embs) )
                
            else:
                self.fill(self.num_elements, self.batch_size)

    #ITERATOR FUNCTIONS
    def __iter__(self):
        return self

    def __next__(self):
        # print(f"iteration { self.current }")
        
        # end of iteration
        if self.current >= self.end:
            raise StopIteration  
            
        else:
            # self.current += self.tokens_batch_size 
            self.current += 1
            #return 
            return self.get_embs(num_tokens=self.tokens_batch_size)

    def __len__(self):
        # return len(self.cached_embs)
        #return total length of iterations, not actual size
        return self.end
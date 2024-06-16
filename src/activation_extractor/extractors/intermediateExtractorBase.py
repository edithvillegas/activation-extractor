import numpy as np
import os, shutil
from collections import OrderedDict

import torch

class IntermediateExtractorBase:
    """
    Extractor for intermediate model outputs.
    Extracts the intermediate calculations (from a specified list of modules) from a pytorch model during inference.  

    :param model: the Pytorch model object
    :param layer_list: list of module names to get outputs from
    :type layer_list: list of strings
    """
    def __init__(self, model, layer_list):
        self.model = model
        self.layer_list = layer_list
        self.hook_handles = {} #store hook handles
        self.intermediate_outputs = {} #store intermediate calculations

    def create_hook(self, layer_name):
        """
        Creates a pytorch hook that saves the output of a given module/layer in the model.
        A pytorch hook is a function that is executed after the module is called.
        
        :param layer_name: name of the module/layer
        :type layer_name: str
        
        :return: the corresponding hook function
        :rtype: function
        """
        def hook(model, input, output):
            self.intermediate_outputs[layer_name] = output
        return hook
    
    def register_hooks(self):
        """
        Registers all the hooks for the specified layers.
        It saves the hook handles to the hook_handles attribute.
        """
        for name, module in self.model.named_modules():
            if name in self.layer_list:
                self.hook_handles[name] = module.register_forward_hook(self.create_hook(name))

    def detach_hooks(self):
        """
        Detaches all the registered hooks saved in the hook_handles attribute.
        """
        for name, hook_handle in self.hook_handles.items():
            hook_handle.remove()

    def clear_all_hooks(self):
        """
        Clears ALL the forward hooks registered to the model. 
        """
        for name, module in self.model.named_modules():
            if name in self.layer_list:
                module._forward_hooks = OrderedDict()

    def get_outputs(self):
        """
        Returns the intermediate activation outputs.
    
        :return: dictionary with intermediate outputs for each specified module/layer.
        :rtype: dictionary
        """
        return self.intermediate_outputs
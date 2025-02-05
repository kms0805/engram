from transformers import AutoModelForCausalLM, AutoConfig
import torch

class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """
        Initializes the task vector from a pretrained and a finetuned checkpoint.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passing in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_checkpoint)
                finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_checkpoint)
                pretrained_state_dict = pretrained_model.state_dict()
                finetuned_state_dict = finetuned_model.state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0, save_path=None):
        """
        Apply a task vector to a pretrained model and optionally save the updated model.
        
        Args:
            pretrained_checkpoint (str): Path or identifier of the pretrained model.
            scaling_coef (float): Scaling coefficient to apply to the task vector.
            save_path (str): Path to save the modified model. If None, the model is not saved.
        
        Returns:
            model: The updated model object.
        """
        with torch.no_grad():
            model = AutoModelForCausalLM.from_pretrained(pretrained_checkpoint)
            pretrained_state_dict = model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector.')
                    continue
                pretrained_state_dict[key] += scaling_coef * self.vector[key]
            model.load_state_dict(pretrained_state_dict, strict=False)
        if save_path:
            model.save_pretrained(save_path)
        return model

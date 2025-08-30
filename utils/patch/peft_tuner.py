from peft.tuners.tuners_utils import BaseTunerLayer

# Define the new function that will replace the original one
def new_set_adapter(self, adapter_names: str | list[str]) -> None:
    if isinstance(adapter_names, str):
        adapter_names = [adapter_names]
    self._active_adapter = adapter_names


# Function to apply the monkey patch
def apply_patch_peft_tuner():
    BaseTunerLayer.set_adapter = new_set_adapter


## Basically, I deleted the following part:

        # # Deactivate grads on the inactive adapter and activate grads on the active adapter
        # for layer_name in self.adapter_layer_names:
        #     module_dict = getattr(self, layer_name)
        #     for key, layer in module_dict.items():
        #         if key in adapter_names:
        #             # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
        #             # happen if a completely different adapter layer is being activated.
        #             layer.requires_grad_(True)
        #         else:
        #             layer.requires_grad_(False)

# I manually add it becasue peft LoRA does not support to train multiple adapters at the same time.
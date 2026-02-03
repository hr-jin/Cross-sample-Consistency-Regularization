from dataclasses import dataclass, field
from sae_lens.saes.sae import SAEMetadata

@dataclass
class CustomSAEConfig:
    model_name: str
    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str

    context_size: int = None
    hook_head_index: int | None = None

    architecture: str = ""
    apply_b_dec_to_input: bool = None
    finetuning_scaling_factor: bool = None
    activation_fn_str: str = ""
    activation_fn_kwargs = {}
    prepend_bos: bool = False
    normalize_activations: str = "none"

    dtype: str = ""
    device: str = ""
    model_from_pretrained_kwargs = {}

    dataset_path: str = ""
    dataset_trust_remote_code: bool = True
    seqpos_slice: tuple = (None,)
    training_tokens: int = -100_000

    sae_lens_training_version: str | None = None
    sae_lens_version: str | None = None
    neuronpedia_id: str | None = None

    metadata: SAEMetadata = field(default_factory=SAEMetadata)

    def __post_init__(self):
        self.metadata.model_name = self.model_name
        self.metadata.sae_lens_training_version = self.sae_lens_training_version
        self.metadata.sae_lens_version = self.sae_lens_version
        self.metadata.hook_name = self.hook_name
        self.metadata.hook_head_index = self.hook_head_index
        self.metadata.prepend_bos = self.prepend_bos
        self.metadata.dataset_path = self.dataset_path
        self.metadata.context_size = self.context_size
        self.metadata.neuronpedia_id = self.neuronpedia_id


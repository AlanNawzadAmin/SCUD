
from omegaconf import OmegaConf

from scud.unet import KingmaUNet
from scud.protein_convnet import ByteNetLMTimeNew

image_nn_name_dict = {
    "KingmaUNet":KingmaUNet,
}

protein_nn_name_dict = {
    "ConvNew": ByteNetLMTimeNew
}

def get_model_setup(cfg, tokenizer=None):
    schedule_conditioning = cfg.model.model in [
        "ScheduleCondition", "DiscreteScheduleCondition",
        "MaskingDiffusion",
    ]
    nn_params = cfg.architecture.nn_params
    nn_params = (OmegaConf.to_container(nn_params, resolve=True)
            if nn_params is not None else {})
    if cfg.architecture.x0_model_class in image_nn_name_dict:
        nn_params = {
            "n_channel": 1 if cfg.data.data == 'MNIST' else 3, 
            "N": cfg.data.N + (cfg.model.model == 'MaskingDiffusion'),
            "n_T": cfg.model.n_T,
            "schedule_conditioning": schedule_conditioning,
            "s_dim": cfg.architecture.s_dim,
            **nn_params
        }
    
        return image_nn_name_dict[cfg.architecture.x0_model_class], nn_params
    
    elif cfg.architecture.x0_model_class in text_nn_name_dict:        
        nn_params["n_T"] = cfg.model.n_T
        nn_params["s_dim"] = cfg.architecture.s_dim
        
        print(nn_params)
        
        nn_params = {
            "config": nn_params,
            "vocab_size": cfg.data.N,#len(tokenizer),
            "schedule_conditioning": schedule_conditioning,
        }
        
        return text_nn_name_dict[cfg.architecture.x0_model_class], nn_params
        
    elif cfg.architecture.x0_model_class in protein_nn_name_dict:
        nn_params = {
            "n_tokens": cfg.data.N + (cfg.model.model == 'MaskingDiffusion'),
            "schedule_conditioning": schedule_conditioning,
            **nn_params
        }
        return protein_nn_name_dict[cfg.architecture.x0_model_class], nn_params

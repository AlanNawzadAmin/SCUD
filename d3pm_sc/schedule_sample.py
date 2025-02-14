import torch
import time

def sample_n_transitions(beta_t, batch_size, times):
    """ For a bunch of betas and times, simulate # transitions before
    time. Repeat batch_size # of times to get [times_dim] + batch_size.
    Note t=0 gives the number of transitions after 1 timestep.
    
    Example usage:
    sample_n_transitions(d3pm.beta_t, 7, torch.tensor([500, 999, 102]))
    >tensor([[ 1.,  6.,  0.],
             [ 0.,  8.,  0.],
             [ 1., 10.,  0.],
             [ 0.,  7.,  0.],
             [ 0.,  3.,  0.],
             [ 0.,  2.,  0.],
             [ 0.,  7.,  0.]], dtype=torch.float64)
    """
    t_shape = times.shape
    times = times.reshape(1, -1)
    beta_t = beta_t.reshape(1, 1, -1).repeat(batch_size, times.shape[1], 1)
    transitions = torch.bernoulli(beta_t)
    transitions = transitions.cumsum(-1)
    transitions = transitions[torch.arange(batch_size)[:, None], torch.arange(times.shape[1])[None, :], times.repeat(batch_size, 1)]
    return transitions.reshape((batch_size,) + t_shape)

def sample_full_transitions(beta_t, batch_size):
    """ For a bunch of betas, simulate # transitions at each timestep.
    
    Example usage:
    sample_n_transitions(d3pm.beta_t, 7)
    > 7 x 1000 tensor
    """
    beta_t = beta_t.reshape(1, -1).repeat(batch_size, 1)
    transitions = torch.bernoulli(beta_t)
    return transitions.bool()

def sample_n_transitions_cont(log_alpha, batch_size, times):
    """ Continuous version of above. alpha is a function that takes t.
    """
    t_shape = times.shape
    times = times.reshape(-1)
    log_alpha_t = log_alpha(times).reshape(1, -1).repeat(batch_size, 1)
    transitions = torch.poisson(-log_alpha_t)
    return transitions
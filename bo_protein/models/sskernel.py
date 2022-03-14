from typing import Union, Optional

import torch

from gpytorch.kernels import Kernel
from gpytorch.constraints import Interval
from gpytorch.utils.memoize import cached
from gpytorch.lazy import NonLazyTensor

from torch import Tensor

class SSKernel(Kernel):
    has_lengthscale = True
    is_stationary = False
    
    def __init__(
        self, 
        max_depth: int, 
        match_decay: Optional[Union[Tensor, float]] = None, 
        gap_decay: Optional[Union[Tensor, float]] = None,
        should_cache: bool = True,
        num_splits: int = 1,
    ):
        """
        This is a reimplementation of the SSK of Moss et al, NIPS '20.

        It assumes unscaled
        """
        super().__init__()
        self.max_depth = max_depth
        self.register_parameter(name="raw_match_decay", parameter=torch.nn.Parameter(torch.ones(1)))
        self.register_parameter(name="raw_gap_decay", parameter=torch.nn.Parameter(torch.ones(1)))
        self.register_constraint("raw_match_decay", Interval(0.0, 1.0))
        self.register_constraint("raw_gap_decay", Interval(0.0, 1.0))
        
        if match_decay is not None:
            self.match_decay = match_decay
        if gap_decay is not None:
            self.gap_decay = gap_decay

        self.should_cache = True
        self.cached_train_matches = None
        self.num_splits = num_splits

    # now set up the 'actual' paramter
    @property
    def match_decay(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_match_decay_constraint.transform(self.raw_match_decay)

    @match_decay.setter
    def match_decay(self, value):
        return self._set_match_decay(value)

    def _set_match_decay(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_match_decay)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_match_decay=self.raw_match_decay_constraint.inverse_transform(value))

    # now set up the 'actual' paramter
    @property
    def gap_decay(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_gap_decay_constraint.transform(self.raw_gap_decay)

    @gap_decay.setter
    def gap_decay(self, value):
        return self._set_gap_decay(value)

    def _set_gap_decay(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_gap_decay)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_gap_decay=self.raw_gap_decay_constraint.inverse_transform(value))
        
    @cached
    def _cached_power_tsr(self, maxlen):
        tril = torch.ones(maxlen, maxlen).tril()
        values = []
        for i in range(maxlen):
            values.append(torch.arange(-i-1.0, maxlen-1-i))
        power_tsr = torch.stack(values)

        power = power_tsr.triu() - torch.diag_embed(power_tsr.diag()) + tril
        new_triu = torch.ones(maxlen, maxlen).triu() - torch.eye(maxlen)
        return power, new_triu
        
    def precalc(self, lengthscale, maxlen):
        power, new_triu = self._cached_power_tsr(maxlen)
        power = power.to(lengthscale)
        new_triu = new_triu.to(lengthscale)
        
        return new_triu * torch.pow(lengthscale * new_triu, power.clamp(min=0))        
    
    def forward(self, x1, x2 = None, diag = False, should_normalize = True, **kwargs):
        """
        we assume as inputs binary masks of size n x L x |a| where |a| is the alphabet size
        """
        if diag:
            return self.lengthscale * torch.ones(x1.shape[-2], dtype=x1.dtype, device=x1.device)
        
        if x1.ndim < 3:
            x1 = x1.unsqueeze(0)

        if x2 is None:
            x2 = x1
            is_symmetric = True
        else:
            is_symmetric = False
        
        # enforce proper devices
        x1 = x1.to(self.gap_decay)
        x2 = x2.to(self.gap_decay)
        
        if self.num_splits > 1:
            x1_old = x1
            x2_old = x2
            
            # print("before: ", x1.shape, x2.shape)
            batch_stacking = x1.shape[-2] // self.num_splits
            x1_list = []
            x2_list = []
            for ind in range(0, x1.shape[-2], batch_stacking):
                x1_curr = x1[..., ind:(ind+batch_stacking), :].unsqueeze(-4)
                x2_curr = x2[..., ind:(ind+batch_stacking), :].unsqueeze(-4)
                if (ind + batch_stacking) >= x1.shape[-2]:
                    # we need to pass falses in to pad
                    x1_zero_match = torch.zeros(
                        *x1_curr.shape[:-2], 
                        batch_stacking - (x1.shape[-2] - ind), 
                        x1_curr.shape[-1], device=x1.device).bool()
                    x2_zero_match = torch.zeros(
                        *x2_curr.shape[:-2], 
                        batch_stacking - (x2.shape[-2] - ind), 
                        x2_curr.shape[-1], device=x2.device).bool()
                    x1_curr = torch.cat((x1_curr, x1_zero_match), dim=-2)
                    x2_curr = torch.cat((x2_curr, x2_zero_match), dim=-2)
                x1_list.append(x1_curr)
                x2_list.append(x2_curr)
            x1 = torch.cat(x1_list, dim=-4)
            x2 = torch.cat(x2_list, dim=-4)
            # print("now: ", x1.shape, x2.shape)
                    
        
        if self.training:
            if self.cached_train_matches is not None:
                matches = self.cached_train_matches
            elif is_symmetric:
                matches = (x1.unsqueeze(-4).unsqueeze(-3) * x2.unsqueeze(-3).unsqueeze(-2)).sum(-1)
        else:
            matches = (x1.unsqueeze(-4).unsqueeze(-3) * x2.unsqueeze(-3).unsqueeze(-2)).sum(-1)        
        
        max_len = matches.shape[-1]
        d_mat = self.precalc(lengthscale=self.gap_decay, maxlen=max_len).unsqueeze(0).unsqueeze(0)
        
        Kp = torch.ones_like(matches)

        for _ in range(self.max_depth - 1):
            Kp = self.match_decay**2 * (matches * Kp)
            Kp = Kp.matmul(d_mat)
            Kp = d_mat.transpose(-1, -2).matmul(Kp)
        Kp = matches * Kp
        kmat = (Kp.sum((-1, -2)) * self.match_decay**2)
        
        # now we normalize
        if should_normalize:
            if is_symmetric:
                kmat_diag = torch.diag_embed(torch.diagonal(kmat, dim1=-2, dim2=-1).clamp(min=1e-10).pow(-0.5))
                norm_kmat = kmat_diag.matmul(kmat).matmul(kmat_diag)
            else:
                if self.num_splits > 1:
                    x1 = x1_old
                    x2 = x2_old
                # if self.num_splits == 1:
                left_norm_term = torch.stack(
                    [self.forward(x, should_normalize=False) for x in x1]
                ).squeeze(-1).squeeze(-1).clamp(min=1e-4)
                right_norm_term = torch.stack(
                    [self.forward(x, should_normalize=False) for x in x2]
                ).squeeze(-1).squeeze(-1).clamp(min=1e-4)

                if self.num_splits > 1:
                    left_norm_term = left_norm_term.transpose(-1, -2)
                    right_norm_term = right_norm_term.transpose(-1, -2)

                # now fix batch cases
                if x1.ndim > 3:
                    left_norm_term = torch.diagonal(left_norm_term, dim1=-2, dim2=-1)
                if x2.ndim > 3:
                    right_norm_term = torch.diagonal(right_norm_term, dim1=-2, dim2=-1)
                
                left_diag = torch.diag_embed(left_norm_term.pow(-0.5))
                right_diag = torch.diag_embed(right_norm_term.pow(-0.5))
                norm_kmat = right_diag.matmul(kmat).matmul(left_diag).transpose(-1, -2)
                # print(torch.linalg.norm(kmat), left_diag.norm(), right_diag.norm())
            if self.num_splits > 1:
                norm_kmat = norm_kmat.sum(-3)
            return self.lengthscale * NonLazyTensor(norm_kmat)
        else:
            return kmat
    
    def __call__(self, x1, x2=None, diag=False, last_dim_is_batch=False, **params):
        return self.forward(x1=x1, x2=x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
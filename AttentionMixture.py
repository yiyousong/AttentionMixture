import io
import copy
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import einops
import random
import diffusers
from mamba_ssm import Mamba
from einops.layers.torch import Rearrange, Reduce
from typing import Union, List
class AttentionMixture(nn.Module):
    '''
    further reduce K and V
    compressing Q using SoftMixture requires S*s*d*G*h, h times higher than modifing K or V.
    only compressing K and V does not change original length (or maybe we should compress Sequence length for all layers by compressing Q?)
    Computation cost: MHSA>>Q to Qcomp=Q@Kcomp=Qcomp@K>K to Kcomp>>Qcomp@Kcomp
    compressing only K is more effiecient.
    '''
    def __init__(self,inputsize=512,decoder_inputsize=None,groupnum=8,headspergroup=16,
                 dim=None,qkdim=128,vdim=128,ffoutsize=None,ffdropout=0.1,attdropout=0.25,latent_length=100,softmaxdim='both',latent_softmax_weight=1):
        super(AttentionMixture, self).__init__()
        self.Rearrange=Rearrange('b d s ->b s d')
        self.latent_length=latent_length
        self.softmaxdim=softmaxdim
        self.latent_softmax_weight=latent_softmax_weight# maybe increase this if you believe many tokens are less important.
        if decoder_inputsize==None:
            decoder_inputsize=inputsize
        if ffoutsize is None:
            ffoutsize=inputsize
        self.attdropout = nn.Dropout(attdropout)  # this is flawed as after dropout weight does not sum to one, but I didn't found a way to efficiently modify tensor inplace
        if dim is not None:
            qkdim,vdim=dim,dim
        self.Wq = nn.Linear(inputsize, headspergroup*groupnum* qkdim)
        self.Wkv = nn.Linear(decoder_inputsize, groupnum* (vdim+qkdim))
        self.qkdim=qkdim
        self.vdim=vdim
        self.norm=RMSNorm(inputsize)
        self.headspergroup=headspergroup
        self.groupnum=groupnum
        self.size = 1 / (qkdim ** 0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.Wz=nn.Sequential(
            RMSNorm(vdim*headspergroup*groupnum),
            nn.Linear(vdim*headspergroup*groupnum,ffoutsize*2),
            nn.SiLU(),
            nn.Dropout(ffdropout),
            RMSNorm(ffoutsize * 2),  # slight less normalization layers may increase performance? I need to test this
            nn.Linear(ffoutsize*2,ffoutsize),
            nn.SiLU(),
            nn.Dropout(ffdropout))
        self.attweight=nn.Sequential(nn.Linear(output_size,latent_length),nn.SiLU(),nn.Linear(latent_length,latent_length))
    def forward(self,x,m=None):

        assert len(x.size())>2
        x=self.norm(x)
        length=x.size(1)
        xsize=list(x.size())
        if m is None:
            m = x
        else:
            assert len(m.size()) >2
        msize=list(m.size())
        qsize=xsize[:-1]+[self.groupnum, self.headspergroup, self.qkdim]
        ksize=msize[:-1]+[self.groupnum, self.qkdim]
        vsize=msize[:-1]+[self.groupnum, self.vdim]
        kv=self.Wkv(m)
        k,v=kv.split([self.qkdim,self.vdim],dim=-1)
        q = self.Wq(x).contiguous().view(qsize)
        weights=self.attweight(k)
        if self.softmaxdim=='both':
            weight1=F.softmax(weights,dim=-1)/length
            weight2=F.softmax(weights,dim=-2)/self.latent_length*self.latent_softmax_weight
            weights=(weight1+weight2)/(1+self.latent_softmax_weight)
        elif self.softmaxdim==-1:
            weights=F.softmax(weights,dim=-1)/length
        elif self.softmaxdim==-2:
            weights=F.softmax(weights,dim=-2)/self.latent_length
        else:
            raise NotImplementedError
        k=einops.einsum(k,weights,'b s d, b s S -> b S d ')
        v=einops.einsum(v,weights,'b s d, b s S -> b S d ')
        k = k.contiguous().view(ksize)
        v = v.contiguous().view(vsize)

        qk=einops.einsum(q,k,'... s g h d,... S g d -> ... g h s S')
        qk = self.softmax(torch.mul(qk, self.size))
        qk=self.attdropout(qk)
        z=einops.einsum(qk,v,'... g h s S,... S g d->... s g h d')
        z=z.flatten(-3)
        out=self.Wz(z)
        if return_attention:
            return out,weight
        else:
            return out

class SoftMixture(nn.Module):
    '''
    this was used to compress speaker embedding
    based on the belief that we can compress tokens by a weight matrix
    '''
    def __init__(self,input_size,output_size=None,num_attlayers=6,latent_length=100, dropout=0.25, d_state=32,softmaxdim='both'):
        super(SoftMixture, self).__init__()
        self.Rearrange=Rearrange('b d s ->b s d')
        # self.model= DimReducMambaBlock(input_size, output_size, num_layers=num_attlayers*2//3, dropout=dropout, d_state=d_state)
        self.model= nn.Linear(input_size, output_size)
        self.attweight=nn.Sequential(nn.Linear(output_size,latent_length),nn.SiLU(),nn.Linear(latent_length,latent_length))
        self.latent_length=latent_length
        self.softmaxdim=softmaxdim
        self.model2=clones(Residuallist([RMSNorm(output_size), Mamba(output_size, d_state=d_state), nn.Dropout(dropout)]),
               num_attlayers // 3)
        self.model=nn.Sequential(*self.model)
        self.model2=nn.Sequential(*self.model2)
    def forward(self,x):
        out=self.model(x)
        length=out.size(1)
        weights=self.attweight(out)
        if self.softmaxdim=='both':
            weight1=F.softmax(weights,dim=-1)*self.latent_length/length
            weight2=F.softmax(weights,dim=-2)
            weights=(weight1+weight2)/2
        elif self.softmaxdim==-1:
            weights=F.softmax(weights,dim=-1)*self.latent_length/length
        elif self.softmaxdim==-2:
            weights=F.softmax(weights,dim=-2)
        else:
            raise NotImplementedError
        out=einops.einsum(out,weights,'b s d, b s S -> b S d ')
        return self.model2(out)
class GQA(nn.Module):
    '''
    Batch first
    '''
    def __init__(self,inputsize=512,decoder_inputsize=None,groupnum=8,headspergroup=16,
                 dim=None,qkdim=128,vdim=128,ffoutsize=None,ffdropout=0.1,attdropout=0.25):
        super(GQA, self).__init__()
        if decoder_inputsize==None:
            decoder_inputsize=inputsize
        if ffoutsize is None:
            ffoutsize=inputsize
        self.attdropout=nn.Dropout(attdropout)
        if dim is not None:
            qkdim,vdim=dim,dim
        self.Wq = nn.Linear(inputsize, headspergroup*groupnum* qkdim)
        self.Wkv = nn.Linear(decoder_inputsize, groupnum* (vdim+qkdim))
        self.qkdim=qkdim
        self.vdim=vdim
        self.norm=RMSNorm(inputsize)
        self.headspergroup=headspergroup
        self.groupnum=groupnum
        self.size = 1 / (qkdim ** 0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.Wz=nn.Sequential(
            RMSNorm(vdim*headspergroup*groupnum),
            nn.Linear(vdim*headspergroup*groupnum,ffoutsize*2),
            nn.SiLU(),
            nn.Dropout(ffdropout),
            nn.Linear(ffoutsize*2,ffoutsize),
            nn.SiLU(),
            nn.Dropout(ffdropout))
    def forward(self,x,m=None,return_attention=False):
        assert len(x.size())>2
        x=self.norm(x)
        xsize=list(x.size())
        if m is None:
            m = x
        else:
            assert len(m.size()) >2
        msize=list(m.size())
        qsize=xsize[:-1]+[self.groupnum, self.headspergroup, self.qkdim]
        ksize=msize[:-1]+[self.groupnum, self.qkdim]
        vsize=msize[:-1]+[self.groupnum, self.vdim]
        kv=self.Wkv(m)
        k,v=kv.chunk(2,dim=-1)
        q = self.Wq(x).view(qsize)
        k = k.contiguous().view(ksize)
        v = v.contiguous().view(vsize)
        qk=einops.einsum(q,k,'... s g h d,... S g d -> ... g h s S')
        qk = self.softmax(torch.mul(qk, self.size))
        qk=self.attdropout(qk)
        z=einops.einsum(qk,v,'... g h s S,... S g d->... s g h d')
        z=z.flatten(-3)
        out=self.Wz(z)
        if return_attention:
            return out,weight
        else:
            return out
class LocalSelfGQA(nn.Module):
    def __init__(self,inputsize=512,decoder_inputsize=None,windowsize=16,groupnum=8,headspergroup=16,
                 dim=None,qkdim=128,vdim=128,ffoutsize=None,ffdropout=0.25,attdropout=0.5,causal=False):
        super(LocalSelfGQA, self).__init__()
        # padding on k and v, this cannot be done on q as the softmax is done on k
        # for causal attention only, pad winsize*2 in front
        # hop_length=window_size/2, for non-causal, the minimum distance would be winsize/4
        # decreasing hop_length may increase the minimum distance but would be hard to code and may increase kv storage size,depends on as_strided function
        # else pad 1.5 winsize in front and 0.5 in end
        # for block-sliding, in each step move half the window size, the first and last half windowsize should be padding as their will be ignored
        self.windowsize=windowsize
        assert windowsize%4==0,'windowsize has to be multiple of 4'
        self.causal=causal
        if decoder_inputsize==None:
            decoder_inputsize=inputsize
        if ffoutsize is None:
            ffoutsize=inputsize
        self.attdropout=nn.Dropout(attdropout) #this is flawed as after dropout weight does not sum to one, but I didn't found a way to efficiently modify tensor inplace
        if dim is not None:
            qkdim,vdim=dim,dim
        self.Wq = nn.Linear(inputsize, headspergroup*groupnum* qkdim)
        self.Wkv = nn.Linear(decoder_inputsize, groupnum* (vdim+qkdim))
        self.qkdim=qkdim
        self.vdim=vdim
        self.norm=RMSNorm(inputsize)
        self.headspergroup=headspergroup
        self.groupnum=groupnum
        self.size = 1 / (qkdim ** 0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.Wz=nn.Sequential(
            RMSNorm(vdim*headspergroup*groupnum),
            nn.Linear(vdim*headspergroup*groupnum,ffoutsize*2),
            nn.SiLU(),
            nn.Dropout(ffdropout),
            RMSNorm(ffoutsize*2),#slight less normalization may increase performance? I need to test this
            nn.Linear(ffoutsize*2,ffoutsize),
            nn.SiLU(),
            nn.Dropout(ffdropout))
    def forward(self,x):
        seqlength=x.size(1)
        drop_last=True if ((seqlength%self.windowsize>self.windowsize//2)or (seqlength%self.windowsize==0)) else False
        x=self.norm(x)
        q=self.Wq(x)
        ifpad,q=pad_to_multiple(q,self.windowsize//2,dim=-2)
        kv=self.Wkv(x)
        _,kv=pad_to_multiple(kv,self.windowsize,dim=-2,mod=self.windowsize//2)
        k,v=kv.chunk(2,dim=-1) #[B,S,D]
        if self.causal:
            k=chunk(F.pad(k,(0,0,self.windowsize//2,0)),self.windowsize)# [B,S/w,2w,d] [[0,k1],[k1,k2],...[...,kn]]
            v=chunk(F.pad(v,(0,0,self.windowsize//2,0)),self.windowsize)# [B,S/w,2w,d] [[0,k1],[k1,k2],...[...,kn]]
            q=einops.rearrange(q,'B (S w) (G H D) -> B G H S w D',G=self.groupnum,H=self.headspergroup,w=self.windowsize//2)
            k=einops.rearrange(k,'B S W (G D) -> B G 1 S W D',G=self.groupnum) # W=2w
            v=einops.rearrange(v,'B S W (G D) -> B G 1 S W D',G=self.groupnum)
            qk=einops.einsum(q,k,' B G H S w D,B G 1 S W D-> B G H S w W')
            qk=qk*self.size
            qk[...,0,:,:self.windowsize//2]=-1e30#droping attention for paddings
            for i in range(self.windowsize//2):
                qk[...,i,self.windowsize // 2+1+i:]=-1e30
            qk=self.softmax(qk,dim=-1)
            qk=self.attdropout(qk)
            qkv=einops.einsum(qk,v,' B G H S w W,B G 1 S W D-> B G H S w D')
            qkv=einops.rearrange(qkv,'B G H S w D->B (S w) (G H D)')
        else:
            k=chunk(F.pad(k,(0,0,self.windowsize//4,self.windowsize//4)),self.windowsize)
            # k=k[...,1:,:,:]# [B,S/w,2w,d] [[0,k1],[k1,k2],...[...,kn]]
            v=chunk(F.pad(v,(0,0,self.windowsize//4,self.windowsize//4)),self.windowsize)# [B,S/w,2w,d] [[0,k1],[k1,k2],...[...,kn]]
            if drop_last:
                k=k[:,:-1,...]
                v=v[:,:-1,...]
            q=einops.rearrange(q,'B (S w) (G H D) -> B G H S w D',G=self.groupnum,H=self.headspergroup,w=self.windowsize//2)
            k=einops.rearrange(k,'B S W (G D) -> B G 1 S W D',G=self.groupnum) # W=2w
            v=einops.rearrange(v,'B S W (G D) -> B G 1 S W D',G=self.groupnum)
            qk=torch.einsum(' B G H S w D,B G h S W D-> B G H S w W',q,k)
            qk=qk*self.size
            qk[...,0,:,:self.windowsize//4]=-1e30
            qk[...,-1,:,-self.windowsize//4:]=-1e30 #droping attention for paddings
            qk=self.softmax(qk)
            qk=self.attdropout(qk)
            qkv=einops.einsum(qk,v,' B G H S w W,B G h S W D-> B G H S w D')
            qkv=einops.rearrange(qkv,'B G H S w D->B (S w) (G H D)')
        if ifpad:
            qkv=qkv[:,:seqlength]
        out = self.Wz(qkv)
        return out



class RMSNorm(nn.Module):
    def __init__(self, d_model, scale = True, dim_cond = None,dim=-1):
        super().__init__()
        self.cond = exists(dim_cond)
        self.to_gamma_beta = nn.Linear(dim_cond, d_model * 2) if self.cond else None
        self.dim=dim
        self.scale = d_model ** 0.5
        if scale:
            if dim == -2 or dim == 1:
                self.gamma = nn.Parameter(torch.ones([d_model,1]))
            else:
                self.gamma = nn.Parameter(torch.ones(d_model))
        else:
            self.gamma=None
    def forward(self, x, cond = None):
        gamma = default(self.gamma, 1)
        out = F.normalize(x, dim = self.dim) * self.scale * gamma
        if not self.cond:
            return out
        assert exists(cond)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim = self.dim)
        gamma, beta = map(lambda t: rearrange(t, 'b d -> b 1 d'), (gamma, beta))
        return out * gamma + beta


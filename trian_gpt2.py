from ast import mod
import dis
from email import generator
import re
from traceback import print_tb
from turtle import mode

from annotated_types import MaxLen
from huggingface_hub import configure_http_backend
from idna import decode
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import time
import inspect
import numpy as np
import os
import multiprocessing as mp
import torch.distributed as dist

class CasulaSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embds%config.n_heads==0
        self.c_attn=nn.Linear(config.n_embds,3*config.n_embds)
        self.c_proj=nn.Linear(config.n_embds,config.n_embds)
        self.n_heads=config.n_heads
        self.n_embds=config.n_embds
        self.c_proj.NANOGPT_SCALE_INIT=1
    
    def forward(self,x):
        b,t,c=x.size()
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embds,dim=2)
        q=q.view(b,t,self.n_heads,c//self.n_heads).transpose(1,2)
        k=k.view(b,t,self.n_heads,c//self.n_heads).transpose(1,2)
        v=v.view(b,t,self.n_heads,c//self.n_heads).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y=y.transpose(1,2).contiguous().view(b,t,c)
        y=self.c_proj(y)
        return y
    
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embds,config.n_embds*4)
        self.gelu=nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(4*config.n_embds,config.n_embds)
        self.c_proj.NANOGPT_SCALE_INIT=1
    
    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x    
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn=CasulaSelfAttention(config)
        self.mlp=MLP(config)
        self.layer1=nn.LayerNorm(config.n_embds)
        self.layer2=nn.LayerNorm(config.n_embds)
    def forward(self,x):
        x=x+self.attn(self.layer1(x))
        x=x+self.mlp(self.layer2(x))
        return x

class GPTConfig:
    n_embds: int = 768
    n_layers: int =12
    n_heads: int =12
    vocab_size: int = 50304
    block_size: int = 1024

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embds),
            wpe=nn.Embedding(config.block_size,config.n_embds),
            h=nn.ModuleList([Block(config) for _ in range (config.n_layers)]),
            layer_norm=nn.LayerNorm(config.n_embds)
        ))
        self.lm_head=nn.Linear(config.n_embds,config.vocab_size)
        self.transformer.wte.weight=self.lm_head.weight
        self.apply(self._init_weight)
    
    def _init_weight(self,module):
        if isinstance(module,nn.Linear):
            std=0.02
            if hasattr(module,'NANOGPT_SCALE_INIT'):
                std*=(2*self.config.n_layers)**-0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
    
    def forward(self,idx,target=None):
        b,t=idx.size()
        assert t<=self.config.block_size ,f"Cannot accept the input of size {t} the vocab size is only {self.config.block_size}"
        pos=torch.arange(0,t,dtype=torch.long,device=idx.device)
        emd=self.transformer.wte(idx)
        pos=self.transformer.wpe(pos)
        x=emd+pos
        for block in self.transformer.h:
            x=block(x)
        logits=self.lm_head(x)
        if target is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),target.view(-1))
        return logits,loss
    
    def configure_optimizer(self,weight_decay,learning_rate):
        param_dict={pn:p for pn,p in self.named_parameters()}
        param_dict={pn: p for pn,p in param_dict.items() if p.requires_grad}
        decay_params=[p for n, p in param_dict.items() if p.dim()>=2]
        nodecay_params=[p for n, p in param_dict.items() if p.dim()<2]
        optim_groups=[
            {'params':decay_params,'weight_decay':weight_decay},
            {'params':nodecay_params,'weight_decay':0.0}
        ]
        num_decay_params=sum(p.numel() for p in decay_params)
        num_nodecay_params=sum(p.numel() for p in nodecay_params)
        fused_available='fused' in inspect.signature(torch.optim.AdamW).parameters
        #use_fused=fused_available and device_type=="cpu"
        optimizer=torch.optim.AdamW(optim_groups,lr=learning_rate,betas=(0.9,0.95), eps=1e-8)
        return optimizer

# class DataLoader():
    
#     def __init__(self,b,t,process_rank,num_processes,split):
#         self.b=b
#         self.t=t
#         self.process_rank=process_rank
#         self.num_processes=num_processes
#         assert split in {'train','val'}

#         data_root='edu_fineweb10B1'
#         shards=os.listdir(data_root)
#         shards=[s for s in shards if split in s]
#         shards=sorted(shards)
#         shards=[os.path.join(data_root,s) for s in shards]
#         self.shards=shards
#         assert len(self.shards)>0 ,f"No shared found for split {split}"
#         self.reset()
#         # with open ('tiny_shakespeare.txt','r') as f:
#         #  text=f.read()
enc=tiktoken.get_encoding('gpt2')
#         # token=enc.encode(text)
#         # self.token=torch.tensor(token)
#         # print(f"Loaded {len(self.token)} token")
#         # print(f"The number of epoch needed will be {len(self.token)//(b*t)}")

#         # self.current_position=0

#     def reset(self):
#         self.current_shard=0
#         self.tokens=load_tokens(self.shards[self.current_position])
#         self.current_position=self.b*self.t*self.process_rank

#     def next_batch(self):
#         b,t=self.b,self.t
#         buf=self.token[self.current_position:self.current_position+b*t+1]
#         x=(buf[:-1]).view(b,t)
#         y=(buf[1:]).view(b,t)
#         self.current_position+=b*t*self.num_processes
#         if self.current_position+(b*t*self.num_processes+1)> len(self.token):
#             self.current_shard=(self.current_shard+1)%len(self.shards)
#             self.tokens=load_tokens(self.shards[self.current_shard])
#             self.current_position=b*t*self.process_rank
#         return x,y
import math
max_lr=6e-4
min_lr=max_lr*0.1
warmup_steps=715
def get_lr(it):
    if it<warmup_steps:
        return max_lr*(it+1)/warmup_steps
    if it>warmup_steps:
        return min_lr
    decay_rate=(it-warmup_steps)/(max_steps-warmup_steps)
    assert 0<=decay_rate<=1
    coeff=0.5*(1.0+math.cos(math.pi*decay_rate))
    return (min_lr+coeff*(max_lr-min_lr))

total_batch_size=524288
B=8
T=1024
assert total_batch_size%(B*T*1)==0 ,"Make sure total batch size is divisible by b*t*8"
grad_accum_steps=total_batch_size//(B*T*1)
#print(f"The grad accum steps is:{grad_accum_steps}")

torch.set_float32_matmul_precision('high')
model=GPT(GPTConfig())
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),eps=1e-8)
max_steps=19073
#optimizer=torch.optim.AdamW(learning_rate=6e-4)

def load_tokens(filename):
    npt=np.load(filename)
    npt=npt.astype(np.int32)
    ppt=torch.tensor(npt,dtype=torch.long)
    return ppt

class DataLoader():
    
    def __init__(self,b,t,num_processes,split):
        self.b=b
        self.t=t
        self.num_processes=num_processes
        assert split in {'train','val'}

        data_root='edu_fineweb10B2'
        shards=os.listdir(data_root)
        shards=[s for s in shards if split in s]
        shards=sorted(shards)
        #print(f"The shared are:{shards}")
        shards=[os.path.join(data_root,s) for s in shards]
        self.shards=shards
        assert len(self.shards)>0 ,f"No shared found for split {split}"
        self.reset()

    def reset(self):
        self.current_shard=0
        #print(f"The current file in use:{self.shards[self.current_shard]}")
        self.tokens=load_tokens(self.shards[self.current_shard])
        self.current_position=self.b*self.t
    
    def next_batch(self,process_rank):
        b,t=self.b,self.t
        self.current_position=self.current_position*process_rank
        buf=self.tokens[self.current_position:self.current_position+b*t+1]
        #print(f"The buf shape is :{buf.shape}")
        x=(buf[:-1]).view(b,t)
        y=(buf[1:]).view(b,t)
        self.current_position+=b*t*self.num_processes
        #print(f"The issue: {self.current_position+(b*t*self.num_processes+1), len(self.tokens)}")
        if self.current_position+(b*t*self.num_processes+1)> len(self.tokens):
            self.current_shard=(self.current_shard+1)%len(self.shards)
            self.tokens=load_tokens(self.shards[self.current_shard])
            self.current_position=b*t*process_rank
            #print(f"The current file is being used is :{self.shards[self.current_shard]}")
        return x,y
    
core_count=1
func=DataLoader(b=4,t=1024,num_processes=core_count,split='train')
process_rank=1

for i in range(max_steps):
        print(f"This is the ith process:{i}")
        t0=time.time()
        last_step=(i==max_steps-1)
        if ((i%250==0 and i>0) and i==last_step):
            model.eval()
            num_return_sequence=4
            max_length=32
            tokens=enc.encode("Hello,I am a language model,")
            tokens=torch.tensor(tokens,dtype=torch.long)
            tokens=torch.unsqueeze(0).repeat(num_return_sequence,1)
            xgen=tokens
            sample_rng=torch.Generator(device="cpu")
            sample_rng.manual_seed(42+process_rank)
            while xgen.size(1)<max_length:
                with torch.no_grad():
                    with torch.autocast(device="cpu"):
                        logits,loss=model(xgen)
                    logits=logits[:,-1,:]
                    probs=F.softmax(logits,dim=-1)
                    topk_probs, topks_indices=torch.topk(probs,50,dim=-1)
                    ix=torch.maltinomial(topk_probs,1,generator=sample_rng)
                    xcol=torch.gather(topk_probs,1,generator=sample_rng)
                    xgen=torch.cat((xgen,xcol),dim=1)
            
            for i in range(num_return_sequence):
                tokens=xgen[i,:max_length].tolist()
                decoded=enc.decode(tokens)
                print(f"The cpu being run:{i}, sample :{decoded}")
        model.train()
        optimizer.zero_grad()
        loss_accum=0
        # print(f"The curren micro step is :{micro_steps}")
        x,y=func.next_batch(process_rank)
        # model.require_backward_grad_sync=(micro_steps==grad_accum_steps-1)
        with torch.autocast(device_type="cpu",dtype=torch.bfloat16):
                logits,loss=model(x,y)
        loss=loss/grad_accum_steps
        loss_accum+=loss.detach()
        loss.backward()
        # world_size=8
        # dist.init_process_group(backend="gloo", rank=process_rank, world_size=world_size)
        # dist.all_reduce(loss_accum,op=dist.ReduceOp.AVG)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        #print(f"The loss for step {i} is :{loss_accum}")

        # lr=get_
        # for param_group in optimizer.param_groups:
        #     param_group['lr']=lr
        # optimizer.step()
        t1=time.time()
        dt=t1-t0
        tokens_processed=func.b*func.t*grad_accum_steps
        token_per_sec=tokens_processed/dt
        #print(f"The loss for step {i} is :{loss_accum} the dt is :{dt}")
        print(f"Step {i:5d} | loss:{loss_accum:.6f} | lr {lr:.42} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok?sec: {token_per_sec:.2f}")


# processes=[]
# core_count=1
# #func=DataLoader(b=4,t=1024,num_processes=core_count,split='train')
# for p in processes:
#         p.join()
# if __name__ == '__main__':
#         for i in range(core_count):
#           #print(f"Using core count:{count}")
#           p=mp.Process(target=main,args=(i,))
#         #print(f"The is the ith process:{i}")
#          #print(f"Using {core_count} cores:")

#           processes.append(p)
#           p.start()
#         #print(f"First 10 squares: {list(result)[:10]}")
        #print(f"The dt will be:{dt}")
        # for p in processes:
        #  p.join()
         
# for i in range (max_steps):
#     t0=time.time()
#     x,y=train_loader.next_batch()
#     optimizer.zero_grad()
#     logits,loss=model(x,y)
#     loss.backward()
#     norm=torch.nn.utils.clip_grad_norm(model.parameter(),1.0)
#     optimizer.step()
#     t1=time.time()
#     dt=(t1-t0)*1000
#     token_per_sec=(train_loader.b*train_loader.t)/(t1-t0)
#     print(f"The loss of the step {i} is {loss.item()} token per sec: {token_per_sec} dt:{dt}")
A block original designed for speaker embedding extraction from my side-project LTCinger
revised a little to compress Attention computation cost (in theory, not tested yet).
compress on K and V, using dual softmax & matmul
![alt text](https://github.com/yiyousong/AttentionMixture/blob/main/AttentionMixture.png?raw=true)

This was never tested. I plan to test this in future when I enter NLP or CV, I don't have any data or card to test this yet. 
uploaded now to provide insight

May 6th update:  
this seems function in my diffusion model. next step is to test on PHI-1.5.
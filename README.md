# CodeCraftAI: Code Generation from Natural Language descriptions


## Dataset Used
[CodeSearchNet](https://github.com/github/CodeSearchNet): A dataset designed for training code generation models.

## Implementations

1. **Seq2Seq**
   - Basic sequence-to-sequence model for converting natural language descriptions to code.
   - *Reference: [arXiv:1409.3215 [cs.CL]](https://arxiv.org/abs/1409.3215)* 

2. **Seq2Seq with Attention**
   - Enhanced sequence-to-sequence model with attention mechanism, allowing the model to focus on different parts of the input sequence during decoding.
   - *Reference: [arXiv:1706.03762 [cs.CL]](https://arxiv.org/abs/1706.03762)* 

3. **FineTuning T5small**
   - Fine-tuned implementation using the transformer-based unified text-to-text model T5small model, leveraging pre-trained language representations for our downstream code generation task.
   - *Reference: [arXiv:1910.10683[cs.LG]](https://arxiv.org/abs/1910.10683)* 

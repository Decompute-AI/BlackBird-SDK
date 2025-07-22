"""Model configuration definitions matching JavaScript ModelDownloader."""

MODEL_CONFIGS = {
    'unsloth/DeepSeek-R1-Distill-Qwen-1.5B': {
        'files': [
          'config.json',
          'tokenizer.json',
          'tokenizer_config.json',
          'generation_config.json',
          'model.safetensors',
          'special_tokens_map.json'
        ]
      },
      'unsloth/Qwen3-0.6B-bnb-4bit': {
        'files': [
          'added_tokens.json',
          'chat_template.jinja',
          'config.json',
          'generation_config.json',
          'merges.txt',
          'model.safetensors',
          'special_tokens_map.json',
          'tokenizer.json',
          'tokenizer_config.json',
          'vocab.json'
        ]
      },
      'unsloth/Qwen3-1.7B-bnb-4bit': {
        'files': [
          'added_tokens.json',
          'chat_template.jinja',
          'config.json',
          'generation_config.json',
          'merges.txt',
          'model.safetensors',
          'special_tokens_map.json',
          'tokenizer.json',
          'tokenizer_config.json',
          'vocab.json'
        ]
      },
      'unsloth/Llama-3.2-1B-Instruct': {
        'files': [
          'config.json',
          'tokenizer.json',
          'tokenizer_config.json',
          'generation_config.json',
          'model.safetensors',
          'special_tokens_map.json'
        ]
      },
      'sentence-transformers/all-MiniLM-L6-v2': {
        'files': [
          'config.json',
          'pytorch_model.bin',
          'special_tokens_map.json',
          'tokenizer_config.json',
          'tokenizer.json',
          'tokenizer.json'
        ]
      },
      'hkunlp/instructor-large': {
        'files': [
          'config.json',
          'pytorch_model.bin',
          'special_tokens_map.json',
          'tokenizer_config.json',
          'tokenizer.json',
          'tokenizer.json'
        ]
      },
      # 'black-forest-labs/FLUX.1-schnell': {
      #   'files': [
      #     #Root level files
      #     'model_index.json',
      #     'ae.safetensors',
      #     'flux1-schnell.safetensors',
      #     'README.md',
      #     '.gitattributes',
      #     'schnell_grid.jpeg',
          
      #     # Scheduler directory
      #     'scheduler/scheduler_config.json',
          
      #     #Text encoder 1 directory
      #     'text_encoder/config.json',
      #     'text_encoder/model.safetensors',
          
      #     # Text encoder 2 directory (with sharded model files)
      #     'text_encoder_2/config.json',
      #     'text_encoder_2/model-00001-of-00002.safetensors',
      #     'text_encoder_2/model-00002-of-00002.safetensors',
      #     'text_encoder_2/model.safetensors.index.json',
          
      #     #Tokenizer 1 directory
      #     'tokenizer/merges.txt',
      #     'tokenizer/special_tokens_map.json',
      #     'tokenizer/tokenizer_config.json',
      #     'tokenizer/vocab.json',
          
      #     # Tokenizer 2 directory
      #     'tokenizer_2/special_tokens_map.json',
      #     'tokenizer_2/spiece.model',
      #     'tokenizer_2/tokenizer.json',
      #     'tokenizer_2/tokenizer_config.json',
          
      #     # Transformer directory
      #     'transformer/config.json',
      #     'transformer/diffusion_pytorch_model-00001-of-00003.safetensors',
      #     'transformer/diffusion_pytorch_model-00002-of-00003.safetensors',
      #     'transformer/diffusion_pytorch_model-00003-of-00003.safetensors',
      #     'transformer/diffusion_pytorch_model.safetensors.index.json',
          
      #     # VAE directory
      #     'vae/config.json',
      #     'vae/diffusion_pytorch_model.safetensors'
      #   ]
      # }
}

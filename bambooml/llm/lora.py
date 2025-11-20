def run(args):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
    except Exception:
        raise RuntimeError('Install transformers, peft, accelerate')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=args.target_modules)
    model = get_peft_model(model, config)
    return {'model': model, 'tokenizer': tokenizer}

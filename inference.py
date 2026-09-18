import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

# 基础模型路径
base_model_path = "Qwen/Qwen3-0.6B"
# 微调后的 adapter 路径
adapter_path = "model/Qwen30.6bhuanhuan"

# 加载基础模型
def load_base_model():
    print("正在加载基础模型...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer

# 加载微调模型
def load_finetuned_model():
    print("正在加载微调模型...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    return model, tokenizer

# 生成回复
def generate_response(model, tokenizer, user_input, system_prompt=None, enable_thinking=False):
    if system_prompt is None:
        system_prompt = "床前明月光"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            streamer=streamer,
        )
    generated_ids = outputs[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response

# 对比两个模型
def compare_models(user_input):
    print("\n" + "="*60)
    print("对比模式：微调模型 vs 基础模型")
    print("="*60)
    print(f"用户输入: {user_input}")
    print("\n--- 微调模型输出 ---")
    model_finetuned, tokenizer_finetuned = load_finetuned_model()
    response_finetuned = generate_response(model_finetuned, tokenizer_finetuned, user_input)
    del model_finetuned, tokenizer_finetuned
    gc.collect()
    torch.cuda.empty_cache()
    print("\n--- 基础模型输出 ---")
    model_base, tokenizer_base = load_base_model()
    response_base = generate_response(model_base, tokenizer_base, user_input)
    del model_base, tokenizer_base
    gc.collect()
    torch.cuda.empty_cache()
    print("\n" + "="*60)
    print("对比完成")
    print("="*60)
    return response_finetuned, response_base


def interactive_mode():
    """交互式对话模式"""
    print("=" * 60)
    print("嬛嬛微调模型 - 交互式对话")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清空对话历史")
    print("输入 'thinking on/off' 开启/关闭思考模式")
    print("输入 'compare' 对比微调模型与基础模型的输出")
    print("=" * 60)

    # 加载微调模型
    model, tokenizer = load_finetuned_model()
    model.eval()
    history = []
    enable_thinking = False

    try:
        while True:
            try:
                user_input = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("再见！")
                break

            if user_input.lower() == "clear":
                history.clear()
                print("对话历史已清空。")
                continue

            if user_input.lower() == "thinking on":
                enable_thinking = True
                print("思考模式已开启。")
                continue

            if user_input.lower() == "thinking off":
                enable_thinking = False
                print("思考模式已关闭。")
                continue

            if user_input.lower() == "compare":
                compare_input = input("请输入要对比的句子: ").strip()
                if compare_input:
                    compare_models(compare_input)
                else:
                    print("输入不能为空。")
                continue

            # 构建消息
            messages = [
                {"role": "system", "content": "你是嬛嬛，来自《甄嬛传》中的角色。请用嬛嬛的语气和说话方式来回答问题。"},
            ]
            for turn in history:
                messages.append(turn)
            messages.append({"role": "user", "content": user_input})

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )

            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            prompt_length = inputs["input_ids"].shape[1]

            print("\n嬛嬛: ", end="", flush=True)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True),
                )

            generated_ids = outputs[0][prompt_length:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # 更新对话历史
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})

            # 保留最近 10 轮对话，避免上下文过长
            if len(history) > 20:
                history = history[-20:]
    finally:
        # 释放模型
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    interactive_mode()

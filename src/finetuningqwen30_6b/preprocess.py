"""数据预处理：将 huanhuan.json 转换为 Qwen3 对话训练格式"""
import json
from pathlib import Path
from transformers import AutoTokenizer

input_path = Path(__file__).parent.parent.parent / "datasets" / "huanhuan.json"
output_path = Path(__file__).parent.parent.parent / "datasets" / "huanhuan_train.jsonl"

SYSTEM_PROMPT = "床前明月光"


def convert():
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"原始数据条数: {len(raw_data)}")

    converted = []
    for item in raw_data:
        user_content = item["instruction"]
        if item.get("input"):
            user_content += "\n" + item["input"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": item["output"]},
        ]
        converted.append({"messages": messages})

    with open(output_path, "w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"转换完成，保存至: {output_path}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    text = tokenizer.apply_chat_template(converted[0]["messages"], tokenize=False)
    print("第一条样本经 chat template 渲染后的文本:")
    print(text)


if __name__ == "__main__":
    convert()

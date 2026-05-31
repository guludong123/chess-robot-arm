import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

from huggingface_hub import HfApi, hf_hub_download
api = HfApi()

output = []

# 获取 Covo-Audio 完整 README
output.append('=== tencent/Covo-Audio-Chat Full README ===')
try:
    readme_path = hf_hub_download('tencent/Covo-Audio-Chat', 'README.md')
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
        output.append(content)
except Exception as e:
    output.append(f'Error: {e}')

# 获取 modeling_covo_audio.py（核心代码）
output.append('\n\n=== modeling_covo_audio.py ===')
try:
    model_path = hf_hub_download('tencent/Covo-Audio-Chat', 'modeling_covo_audio.py')
    with open(model_path, 'r', encoding='utf-8') as f:
        content = f.read()[:3000]  # 前3000字符
        output.append(content)
except Exception as e:
    output.append(f'Error: {e}')

# 获取 config.json
output.append('\n\n=== config.json ===')
try:
    config_path = hf_hub_download('tencent/Covo-Audio-Chat', 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
        output.append(content)
except Exception as e:
    output.append(f'Error: {e}')

# GitHub 仓库
output.append('\n\n=== GitHub Link ===')
output.append('https://github.com/Tencent/Covo-Audio')

with open('f:/chessrobotarm/covo_audio_details.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print('Done! Saved to covo_audio_details.txt')
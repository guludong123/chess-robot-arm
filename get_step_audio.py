import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

from huggingface_hub import HfApi, hf_hub_download, list_models
api = HfApi()

output = []

# Step-Audio-2-mini 详细信息
output.append('=== Step-Audio-2-mini 详细信息 ===')

model_id = 'stepfun-ai/Step-Audio-2-mini'

try:
    info = api.model_info(model_id)
    output.append(f'Model ID: {info.id}')
    output.append(f'Pipeline: {info.pipeline_tag}')
    output.append(f'Tags: {info.tags}')
    output.append(f'Downloads: {info.downloads}')
    output.append(f'Likes: {info.likes}')
    output.append(f'Library: {info.library_name}')
    output.append(f'Created: {info.created_at}')
    output.append(f'Last Modified: {info.last_modified}')

    output.append('\n--- Model Files ---')
    for f in info.siblings:
        output.append(f'  {f.rfilename}')

except Exception as e:
    output.append(f'Error: {e}')

# 获取完整README
output.append('\n--- Full README ---')
try:
    readme_path = hf_hub_download(model_id, 'README.md')
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
        output.append(content)
except Exception as e:
    output.append(f'Error: {e}')

# 获取 config.json
output.append('\n--- config.json ---')
try:
    config_path = hf_hub_download(model_id, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
        output.append(content)
except Exception as e:
    output.append(f'Error: {e}')

# 获取 modeling 文件（如果有）
output.append('\n--- 检查 modeling_step_audio_2.py ---')
try:
    model_path = hf_hub_download(model_id, 'modeling_step_audio_2.py')
    with open(model_path, 'r', encoding='utf-8') as f:
        content = f.read()[:3000]
        output.append(content)
except Exception as e:
    output.append(f'Error: {e}')

# 搜索 Step-Audio 相关的其他模型
output.append('\n\n=== Step-Audio 系列所有模型 ===')
try:
    models = list_models(author='stepfun-ai', limit=30)
    for m in models:
        tags_str = ', '.join(m.tags[:8]) if m.tags else ''
        output.append(f'{m.id} | pipeline: {m.pipeline_tag} | downloads: {m.downloads} | tags: {tags_str}')
except Exception as e:
    output.append(f'Error: {e}')

# 写入文件
with open('f:/chessrobotarm/step_audio_details.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print('Done! Saved to step_audio_details.txt')
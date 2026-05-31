import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

from huggingface_hub import HfApi, list_models, hf_hub_download
api = HfApi()

output = []

# 详细检查发现的关键模型
key_models = [
    'LiquidAI/LFM2.5-Audio-1.5B',       # Liquid AI 音频模型
    'KRAFTON/Raon-SpeechChat-9B',       # KRAFTON 9B语音对话
    'kyutai/hibiki-zero-3b-pytorch-bf16', # Kyutai 多语言
    'stepfun-ai/Step-Audio-2-mini',      # StepFun（已确认中文）
]

output.append('=== 详细检查关键模型 ===')

for model_id in key_models:
    output.append(f'\n--- {model_id} ---')
    try:
        info = api.model_info(model_id)
        output.append(f'Pipeline: {info.pipeline_tag}')
        output.append(f'Tags: {info.tags}')
        output.append(f'Downloads: {info.downloads}')
        output.append(f'Library: {info.library_name}')
        tags_lower = [t.lower() for t in (info.tags or [])]
        has_zh = 'zh' in tags_lower or 'chinese' in tags_lower
        output.append(f'Chinese support: {"YES" if has_zh else "NO"}')

        # 获取README
        try:
            readme_path = hf_hub_download(model_id, 'README.md')
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()[:2000]
                output.append(f'README preview:\n{content[:1000]}')
        except Exception as e:
            output.append(f'README: {e}')

    except Exception as e:
        output.append(f'Error: {e}')

# 搜索更多中国公司的语音模型
output.append('\n\n=== 搜索中国公司语音模型 ===')
chinese_companies = ['alibaba', 'baidu', 'bytedance', 'xiaomi', 'huawei', 'tencent', 'stepfun', 'minimax', 'moonshot']

for company in chinese_companies:
    output.append(f'\n--- {company} ---')
    try:
        models = list_models(author=company, limit=20)
        for m in models:
            tags_lower = [t.lower() for t in (m.tags or [])]
            is_audio = any(t in tags_lower for t in ['audio', 'speech', 'voice', 'tts', 'asr'])
            if is_audio:
                has_zh = 'zh' in tags_lower or 'chinese' in tags_lower
                zh_marker = ' [中文]' if has_zh else ''
                output.append(f'  {m.id}{zh_marker} | pipeline: {m.pipeline_tag} | downloads: {m.downloads}')
    except Exception as e:
        output.append(f'  Error: {e}')

# 搜索ModelScope镜像（国内平台）
output.append('\n\n=== ModelScope 可能的中文语音模型 ===')
output.append('ModelScope是国内平台，可能有更多中文语音模型：')
output.append('https://modelscope.cn/models?search=speech-to-speech')
output.append('https://modelscope.cn/models?search=audio-to-audio')
output.append('https://modelscope.cn/models?search=语音对话')

# 写入文件
with open('f:/chessrobotarm/native_chinese_voice_models.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print('Done!')
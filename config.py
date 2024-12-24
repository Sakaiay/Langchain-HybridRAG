# config


# Embedding 配置
EMBEDDING_CONFIG = {
    "model": "bge-m3",
    "params": {
        "base_url": "http://127.0.0.1:9997/v1",
        "api_key": "EMPTY",
    },
}
# LLM 配置
# LLM_CONFIG = {
#     "model": "qwen2-instruct",
#     "params":{
#         "model_name" : "qwen2-instruct",
#         "temperature": 0.1,
#         "base_url": 'http://127.0.0.1:9997/v1',
#         "api_key": 'EMPTY',
#     }
# }

LLM_CONFIG = {
    "model": "Qwen2.5-14B-Instruct",
    "params":
        {
        "model_name" : "Qwen2.5-14B-Instruct",
        "temperature": 0.1,
        "base_url": 'http://localhost:5551/v1',
        "api_key": 'EMPTY'
        }
    }

def get_config():
    return {
        "embedding": EMBEDDING_CONFIG,
        "llm": LLM_CONFIG
    }
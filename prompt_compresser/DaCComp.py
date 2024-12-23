from openai import OpenAI
import re
import json
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor


prompt_format_local = \
"""# 指令

给你一个文本和一个问题，你需要输出以下内容：
- 分析：分析出哪些文本中与问题相关，给出理由
- 源相关文本：直接对上述分析的与问题相关的部分文本原文提取，若没有则为'无'
- 摘要：对源相关文本进行总结，确保摘要直接不包含任何代词，且包含回答问题所需的一切细节。不要进行假设或试图回答问题，您的工作只是进行总结。若没有则为'无'。

请以json格式输出


# 文本

{text}

# 问题

{query}

# 输出

"""


prompt_format_global = \
"""# 指令

给你如下的文本和一个问题，根据文本回答问题。

# 文本

{text}

# 问题

{query}

# 输出

"""


class DaCComp:
    
    """
    分治法压缩 Divide and Conquer Compression (DaCComp):
    - 第一阶段对每个RAG文档单独并行提取摘要，
    - 第二阶段拼接所有摘要作为参考文档，并回答问题。
    """
    
    def __init__(self):
        pass
    
    def parse_json_output(self, text):
        """
        解析大模型的json输出
        """
        if '```' in text:
            text = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL).group(1)

        parsed_output = json.loads(text)
        
        return parsed_output
    
    def compress_docs(self, query, docs, llm, num_thread=4):
        prompt_local = [prompt_format_local.format(text=doc, query=query) for doc in docs]
        with ThreadPoolExecutor() as executor:
            rets = list(executor.map(llm.invoke, prompt_local))
            
        rets = [self.parse_json_output(item) for item in rets]
        summary_local = [item['摘要'] for item in rets]
        return '\n\n'.join(summary_local)
    
        
    def compress_and_answer(self, query, docs, llm, num_thread=4):
        compressed_text = self.compress_docs(query, docs, llm, num_thread)
        prompt_global = prompt_format_global.format(text=compressed_text, query=query)
        ans = llm.invoke(prompt_global)
        return ans
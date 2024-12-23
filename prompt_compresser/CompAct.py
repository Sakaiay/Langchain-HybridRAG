from openai import OpenAI
import re
import json
from pprint import pprint

prompt_compress_1st = \
"""1. 给你一个问题和源文档，你需要总结源文档中与问题相关的部分。确保摘要简洁直接，且不包含任何代词。不要进行假设或试图回答问题；您的工作只是进行总结。
2. 仅根据摘要信息对其进行评估，
- 如果摘要包含了基本信息以初步回答问题，但是可以考虑继续加入新信息，以更好的回答问题，那应该评估为“可继续完善”。
- 如果摘要中缺少回答问题所需的足够细节，那应该评估为“不完整”。
- 如果摘要提供了所有必要的细节，那应该评估为“完整”。
你需要以json格式输出:
{{
    "摘要":XX,
    "评估分析":XX,
    "评估结果":XX
}}

问题：{query}

源文档：{text}"""

prompt_compress_non_1st = \
"""1. 给你一个问题、上一轮的摘要、上一轮的的评估和一段新的源文档，你需要总结源文档中与问题相关的部分并与上一轮的摘要合并。评估指出回答问题所需的缺失信息。确保摘要简洁直接，且不包含任何代词。不要进行假设或试图回答问题；您的工作只是进行总结。
2. 仅根据摘要信息对其进行评估，
- 如果摘要包含了基本信息以初步回答问题，但是可以考虑继续加入新信息，以更好的回答问题，那应该评估为“可继续完善”。
- 如果摘要中缺少回答问题所需的足够细节，那应该评估为“不完整”。
- 如果摘要提供了所有必要的细节，那应该评估为“完整”。
你需要以json格式输出:
{{
    "摘要":XX,
    "评估分析":XX,
    "评估结果":XX
}}

问题：{query}

源文档：{text}

先前的摘要：{prev_summary}

先前的评估：{prev_eval}"""


def call_LLM(client:OpenAI, model_name, prompt):
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是一个优秀的助理。"},
            {"role": "user", "content": prompt},
        ]
    )
    ret = chat_response.choices[0].message.content
    return ret

class CompAct:
    
    """
    参考：COMPACT: Compressing Retrieved Documents Actively for Question Answering. https://arxiv.org/pdf/2407.09014
    他的思路分为两部分：
    - 持续压缩：假设top-k设为10，窗口长度设为5，每次取5个文档，拼成材料让大模型生成摘要，同时并判断是否材料足够完整以回答问题，以一个特殊字段来标识，只有两类：[完整]和[不完整]。
    - 早停：如果某次评估认为材料足够完整了，跳出循环
    
    在原来对摘要对问题的完整性上增加一个新的分类：可继续完善。如果摘要包含了基本信息以初步回答问题，但是可以考虑继续加入新信息，以更好的回答问题，那应该评估为“可继续完善”。这样停止早停，应该会把所有文档都会遍历。
    """

    def __init__(self):
        pass

    def create_prompt(self, query, text, iter_idx, prev_summary='', prev_eval=''):
        """
        持续压缩的prompt有两个，分别是第一次的（没有先前的摘要和评估）和非第一次的（需要拼上先前的摘要和评估）
        """
        if iter_idx == 0:
            prompt_template = prompt_compress_1st
            prompt = prompt_template.format(
                query=query,
                text=text,
            )
        else:
            prompt_template = prompt_compress_non_1st
            prompt = prompt_template.format(
                query=query,
                text=text,
                prev_summary=prev_summary,
                prev_eval=prev_eval,
            )
        return prompt

    def parse_json_output(self, text):
        """
        解析大模型的json输出
        """
        if '```' in text:
            text = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL).group(1)

        parsed_output = json.loads(text)
        
        return parsed_output

    def compress_docs(self, query, docs, client, model_name, doc_batch_size=3):
        """
        将RAG的文档压缩
        """
        prev_summary = []
        prev_eval = []
        prompt_history = []

        for iter_idx, doc_indice in enumerate(range(0, len(docs), doc_batch_size)):

            prev_summary_temp = prev_summary[-1] if iter_idx!= 0 else ""
            prev_eval_temp = prev_eval[-1] if iter_idx!= 0 else ""

            a_batch_docs = docs[doc_indice : doc_indice + doc_batch_size]
            a_doc_text = "\n".join(a_batch_docs)
            a_prompt = self.create_prompt(
                query = query,
                text = a_doc_text,
                iter_idx = iter_idx,
                prev_summary = prev_summary_temp,
                prev_eval = prev_eval_temp
            )

            llm_ret = call_LLM(client, model_name, a_prompt)
            parse_llm_ret = self.parse_json_output(llm_ret)

            print(iter_idx)
            pprint(parse_llm_ret, width=200)

            prev_summary.append(parse_llm_ret['摘要'])
            prev_eval.append(parse_llm_ret['评估分析'])
            prompt_history.append(a_prompt)

            if parse_llm_ret['评估结果'] == '完整':
                break

        return prev_summary[-1], prev_summary, prev_eval, prompt_history

if __name__ == '__main__':

    # 读取数据
    data = []
    file_path = 'data/data_>=10.json'
    with open(file_path) as f:
        for line in f:
            a_record = json.loads(line)
            data.append(a_record)

    print(len(data))

    # 准备大模型
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:9997/v1"
    model_name = 'qwen2-instruct'
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # 测试效果
    doc_batch_size = 1
    data_indice = 1
    query = data[data_indice]['query']

    query = '90版本阿修罗武器排行，给出具体的理由'
    docs = data[data_indice]['pos']

    print(query)

    compact = CompAct()

    summary, prev_summary, prev_eval, prompt_history = compact.compress_docs(query, docs, client, model_name, doc_batch_size)



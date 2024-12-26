import re
import yaml
import json
import openai
import traceback
import numpy as np
from pysbd import Segmenter
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List
from langchain_openai.chat_models import ChatOpenAI
from langchain.embeddings.base import Embeddings


class Metric_With_LLM_Interface(ABC):
    """调用大模型的基类"""
    def __init__(self, llm:ChatOpenAI=None):
        """
        初始化评估器
        
        :param llm: 用于评估检索内容的 LLM 模型实例
        """
        self.llm = llm
        self.sentence_segmenter = Segmenter()

    def call_llm(self, prompt: str) -> str:
        assert self.llm is not None, "llm is not set"
        msg = self.llm.invoke(prompt)
        content = msg.content
        return content

    def parse_json_output(self, text):
        """
        解析大模型的json输出
        """
        if '```' in text:
            text = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL).group(1)

        parsed_output = json.loads(text)
        
        return parsed_output
    
    @abstractmethod
    def evaluate(self, instance:Dict) -> float:
        pass
    

class Metric_With_Embedding_Interface(ABC):
    """调用embedding模型的基类"""
    def __init__(self, embedding_interface:openai.Client=None):
        """
        初始化评估器
        
        :param embedding_interface: 用于评估检索内容的 embedding 模型实例
        """
        self.embedding_interface = embedding_interface

    def get_embeddings(self, sentences):
        assert type(sentences) == list
        assert self.embedding_interface is not None, "embedding is not set"
        em_params = {'model': 'bge-m3', "input": sentences}
        msg = self.embedding_interface.embeddings.create(**em_params)
        embed = [item.embedding for item in msg.data]
        return np.array(embed)

    def compute_similarity(self, sentence_list_a, sentence_list_b):

        embeddings1 = self.get_embeddings(sentence_list_a)
        embeddings2 = self.get_embeddings(sentence_list_b)
        
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        return similarity_matrix


class Faithfulness(Metric_With_LLM_Interface):
    """该指标从回答中抽取原子化的陈述，并依次判断陈述是否可由上下文得出。分数为可由上下文得出的陈述的比例"""
    def __init__(self, llm):
        super(Faithfulness, self).__init__(llm)
        self.prompt_4_create_statements = \
"""# 指令

给定一个问题、一个包含若干句子的回答，分析每个句子并将每个句子分解为一个或多个原子化的完全可理解的内容陈述，同时确保每个语句中不使用代词，且每个对象都需要具体，在其所在陈述中可理解。以JSON格式输出。

# 格式参考

回答格式形如：
[
    {{
        "句子编号":1,
        "句子内容":"原始内容"
    }},
    ...
]

输出格式形如：
[
    {{
        "句子编号":1,
        "原子化的内容陈述":[
            "陈述1",
            "陈述2",
            ...
        ]
    }},
    ...
]

请按照上述格式

# 问题

{query}

# 回答

{answer}

# 输出
"""
        self.prompt_4_verdict_statements = \
"""# 指令

你的任务是根据给定的上下文判断一系列内容陈述的可信度。对于每个陈述，如果可以根据上下文直接推断该出陈述，则返回1，否则返回0。

# 格式参考

陈述格式形如：
[
    {{
        "陈述编号":1,
        "陈述内容":"原始内容"
    }},
    ...
]

输出格式形如：
[
    {{
        "陈述编号":1,
        "陈述内容":"原始内容",
        "陈述是否可以由上下文推断的思考":"思考内容",
        "陈述可信度":0
    }},
    ...
]

请按照上述格式

# 上下文

{context}

# 陈述

{statements}

# 输出
"""

    def create_statements(self, instance:Dict):
        answer, query = instance["response"], instance["user_input"]
        sentences = self.sentence_segmenter.segment(answer)
        sentences_with_index = [
            {
                "陈述编号":indice,
                "陈述内容":sentence.strip(),
            }
            for indice, sentence in enumerate(sentences)
        ]
        prompt = self.prompt_4_create_statements.format(
            query = query,
            answer = json.dumps(sentences_with_index, indent=2, ensure_ascii=False)
        )
        ret = self.call_llm(prompt)
        ret = metric.parse_json_output(ret)
        statements = sum([item['原子化的内容陈述'] for item in ret], [])
        return statements

    def create_verdicts(self, instance:Dict, statements:List):
        contexts_str = "\n".join(instance["retrieved_contexts"])
        statements_with_index = [
            {
                "句子编号":indice,
                "句子内容":a_statement.strip(),
            }
            for indice, a_statement in enumerate(statements)
        ]
        statements_str = json.dumps(statements_with_index, indent=2, ensure_ascii=False)
        prompt = self.prompt_4_verdict_statements.format(
            context = contexts_str,
            statements = statements_str
        )
        ret = self.call_llm(prompt)
        statement_verdicts = metric.parse_json_output(ret)
        return statement_verdicts
    
    def compute_score(self, statement_verdicts):
        num_faithful_statements = sum([item['陈述可信度'] for item in statement_verdicts])
        num_statements = len(statement_verdicts)
        if num_statements:
            score = num_faithful_statements / num_statements
        else:
            score = np.nan
        return score

    def evaluate(self, instance:Dict) -> float:

        try:
            statements = self.create_statements(instance)
        except Exception as e:
            print("生成陈述|发生错误:")
            traceback.print_exc()
            print(f'当前数据:\n{json.dumps(instance, indent=2, ensure_ascii=False)}')
            return np.nan
        
        if len(statements)==0 or statements is None:
            print('陈述生成异常')
            print(f'当前数据:\n{json.dumps(instance, indent=2, ensure_ascii=False)}')
            return np.nan
        
        try:
            statement_verdicts = self.create_verdicts(instance, statements)
        except Exception as e:
            print("判断陈述|发生错误:")
            traceback.print_exc()
            print(f'当前数据:\n{json.dumps(instance, indent=2, ensure_ascii=False)}')
            return np.nan
            
        if len(statement_verdicts)==0 or statement_verdicts is None:
            print('陈述判断异常')
            print(f'当前数据:\n{json.dumps(instance, indent=2, ensure_ascii=False)}')
            return np.nan
            
        try:
            score = self.compute_score(statement_verdicts)
            return score
        except Exception as e:
            print("计算分数|发生错误:")
            traceback.print_exc()
            print(f'当前数据:\n{json.dumps(instance, indent=2, ensure_ascii=False)}')
            return np.nan


class Answer_Relevance(Metric_With_LLM_Interface, Metric_With_Embedding_Interface):
    """该指标根据回答生成多个问题，并计算生成的问题与实际的问题的语义相似度"""
    def __init__(self, llm:ChatOpenAI, embedding:Embeddings):
        Metric_With_LLM_Interface.__init__(self, llm)
        Metric_With_Embedding_Interface.__init__(self, embedding)
        self.prompt_4_answer_relevance = \
"""# 指令

为给定的回答生成一个问题，并识别回答是否为含糊的。如果回答是含糊的，则将含糊的记为1，如果回答是肯定的，则记为0。含糊的的回答是指回避、含糊或模棱两可的回答。例如，“我不知道”或 “我不确定”就是含糊的的回答。
以如下json格式输出：
{{
    "生成的问题":"问题内容",
    "回答对于生成的问题来说是否含糊(分析)":"分析内容",
    "回答对于生成的问题来说是否含糊(结果)":0/1,
}}

# 回答

{answer}

# 输出

"""

    def create_query(self, instance:Dict, num_gen:int=10):
        answer = instance["response"]
        prompt = self.prompt_4_answer_relevance.format(answer=answer)
        query_gen = []
        for _ in range(num_gen):
            ret = self.call_llm(prompt)
            ret = self.parse_json_output(ret)
            query_gen.append(ret)
        return query_gen
    
    def eval_query_similarity(self, instance:Dict, query_gen:List):
        query_real = instance['user_input']
        query_gen_list = [item['生成的问题'] for item in query_gen]
        query_relevance = [item['回答对于生成的问题来说是否含糊(结果)'] for item in query_gen]

        cosine_sim = self.compute_similarity([query_real], query_gen_list)
        query_relevance = np.array([query_relevance])
        scores = cosine_sim * (1-query_relevance)
        aveaged_score = np.mean(scores)

        return aveaged_score
    
    def evaluate(self, instance):
        try:
            query_gen = self.create_query(instance)
        except Exception as e:
            print("生成问题|发生错误:")
            traceback.print_exc()
            print(f'当前数据:\n{json.dumps(instance, indent=2, ensure_ascii=False)}')
            return np.nan
        
        if len(query_gen)==0 or query_gen is None:
            print('问题生成异常')
            print(f'当前数据:\n{json.dumps(instance, indent=2, ensure_ascii=False)}')
            return np.nan
        
        try:
            score = self.eval_query_similarity(instance, query_gen)
            return score
        except Exception as e:
            print("计算问题相似度|发生错误:")
            traceback.print_exc()
            print(f'当前数据:\n{json.dumps(instance, indent=2, ensure_ascii=False)}')
            return np.nan


class Context_Precision(Metric_With_LLM_Interface):
    """该指标计算上下文整体对回答的有效性，结合排名"""
    def __init__(self, llm):
        super(Context_Precision, self).__init__(llm)

        self.prompt_4_answer_usefulness = \
"""# 指令

给出问题、答案和上下文，验证上下文是否有助于得出给定的答案。你需要先分析内容作为判决理由，如果有用，则判决结果为1，如果没用，则判决结果为0。
输出格式如下：
{{
    "判决理由":"对内容的分析和思考",
    "判决结果":0/1
}}

# 问题

{query}

# 回答

{answer}

# 上下文

{context}

# 输出
"""

    def verdict_usefulness(self, instance:Dict):
        query, answer, retrieved_contexts = instance['user_input'], instance['response'], instance['retrieved_contexts']

        prompts = [
            self.prompt_4_answer_usefulness.format(
                query = query,
                answer = answer,
                context = context,
            )
            for context in retrieved_contexts
        ]
        rets = [self.call_llm(item) for item in prompts]
        rets = [self.parse_json_output(item) for item in rets]
        return rets
    
    def calculate_average_precision(self, scores):
        final_score = np.nan

        denominator = sum(scores)
        if denominator <=0:
            return 0
        numerator = sum(
            [
                (sum(scores[: i + 1]) / (i + 1)) * scores[i]
                for i in range(len(scores))
            ]
        )
        final_score = numerator / denominator
        return final_score


    def evaluate(self, instance):
        try:
            verdict_dict = self.verdict_usefulness(instance)
            verdict_score = [item['判决结果'] for item in verdict_dict]
        except Exception as e:
            print("计算问题相似度|发生错误:")
            traceback.print_exc()
            print(f'当前数据:\n{json.dumps(instance, indent=2, ensure_ascii=False)}')
            return np.nan
        try:
            average_precision = self.calculate_average_precision(verdict_score)
            return average_precision
        except Exception as e:
            print("计算分数|发生错误:")
            traceback.print_exc()
            print(f'当前数据:\n{json.dumps(instance, indent=2, ensure_ascii=False)}')
            return np.nan


class Context_Recall(Metric_With_LLM_Interface):
    """该指标计算回答的每句话是否与上下文相关，取平均相关度"""
    def __init__(self, llm):
        super(Context_Recall, self).__init__(llm)
        self.prompt_4_sentence_contextual_relevance = \
"""# 指令

给定上下文和回答，分析回答中的每个句子，并判断该句子是否与给定的上下文相关,使用 1 或 0 进行二进制分类。
注意，非涉及具体内容的句子，例如过渡句，不需要分析，直接跳过。
输出格式如下：
[
    {{
        "句子":"回答中第1个句子"
        "判决理由":"对句子是否与给定的上下文相关的分析和思考",
        "判决结果":0/1
    }},
    ...
]

# 回答

{answer}

# 上下文

{context}

# 输出
"""

    def verdict_sentence_contextual_relevance(self, instance:Dict):

        answer, retrieved_contexts = instance['response'], instance['retrieved_contexts']

        prompts = [
            self.prompt_4_sentence_contextual_relevance.format(
                answer = answer,
                context = context
            )
            for context in retrieved_contexts
        ]
        rets = [self.call_llm(item) for item in prompts]
        rets = [self.parse_json_output(item) for item in rets]
        return rets
    
    def compute_precision(self, scores):
        denom = len(scores)
        numerator = sum(scores)
        ave_score = numerator / denom if denom > 0 else np.nan
        return ave_score
    
    def evaluate(self, instance):
        try:
            sentence_contextual_relevance_verdicts = self.verdict_sentence_contextual_relevance(instance)
        except Exception as e:
            print("生成判决|发生错误:")
            traceback.print_exc()
            print(f'当前数据:\n{json.dumps(instance, indent=2, ensure_ascii=False)}')
            return np.nan
        
        try:
            context_scores = []
            for a_context_verdicts in sentence_contextual_relevance_verdicts:
                tmp_scores = [item['判决结果'] for item in a_context_verdicts]
                context_scores.extend(tmp_scores)   
        except Exception as e:
            print("判决分数解析|发生错误:")
            traceback.print_exc()
            print(f'当前数据:\n{json.dumps(instance, indent=2, ensure_ascii=False)}')
            return np.nan
        
        try:
            ave_score = self.compute_precision(context_scores)
            return ave_score
        except Exception as e:
            print("计算分数|发生错误:")
            traceback.print_exc()
            print(f'当前数据:\n{json.dumps(instance, indent=2, ensure_ascii=False)}')
            return np.nan


class Context_Precision(Metric_With_LLM_Interface):
    """给事实陈述和回答，计算回答陈述的准确率、召回率、F1分数和假阴性率"""
    def __init__(self, llm):
        super(Context_Precision, self).__init__(llm)
        self.prompt_4_answer_referred_classify = \
"""
给定一个事实陈述和一个答案，分析答案的每个语句，并判断该语句是否得到事实陈述的支持，若得到支持则判断为1，若没有得到支持则判断为0。
输出格式如下：
[
    {{
        "答案语句":"答案的第1个语句",
        "判断分析":"该语句是否能得到事实陈述中任意一句话的支持",
        "判断结果":0/1
    }},
    ...
]

# 回答

{answer}

# 事实陈述

{reference}

# 输出
"""

        self.prompt_4_ground_truth_referred_classify = \
"""
给定一个事实陈述和一个答案，分析事实陈述的每个语句，并判断该语句是否在答案中体现出来，若存在则判断为1，若不存在则判断为0。
输出格式如下
[
    {{
        "事实陈述语句":"事实陈述的第1个语句",
        "判断分析":"分析该语句是否在答案中体现出来",
        "判断结果":0/1
    }},
    ...
]

# 回答

{answer}

# 事实陈述

{reference}

# 输出
"""

    def classify_answer_referred(self, instance:Dict):

        answer, ground_truth = instance['response'], instance['reference']
        answer_sentences = self.sentence_segmenter.segment(answer)

        prompt = self.prompt_4_answer_referred_classify.format(
            answer = json.dumps(answer, indent=2, ensure_ascii=False),
            reference = ground_truth
        )
        ret = self.call_llm(prompt)
        ret = self.parse_json_output(ret)
        TP, FP = [], []
        for item in ret:
            tmp = {
                "语句":item["答案语句"],
                "分析":item["判断分析"]
            }
            if item["判断结果"] == 1:
                TP.append(tmp)
            elif item["判断结果"] == 0:
                FP.append(tmp)
        return TP, FP
    
    def classify_ground_truth_referred(self, instance:Dict):

        answer, ground_truth = instance['response'], instance['reference']
        ground_truth_sentences = self.sentence_segmenter.segment(ground_truth)

        prompt = self.prompt_4_ground_truth_referred_classify.format(
            answer = answer,
            reference = json.dumps(ground_truth, indent=2, ensure_ascii=False)
        )
        ret = self.call_llm(prompt)
        ret = self.parse_json_output(ret)

        FN = []
        for item in ret:
            tmp = {
                "语句":item["事实陈述语句"],
                "分析":item["判断分析"]
            }
            if item["判断结果"] == 0:
                FN.append(tmp)
        
        return FN
    
    def calculate_metrics(self, TP, FP, FN):

        if TP + FP != 0:
            precision = TP / (TP + FP)
        else:
            precision = 0  
            
        if TP + FN != 0:
            recall = TP / (TP + FN)
        else:
            recall = 0  
            
        if precision + recall != 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0  
            
        if TP + FN != 0:
            fnr = FN / (FN + TP)
        else:
            fnr = 0  
        
        return precision, recall, f1_score, fnr
    
    def evaluate(self, instance):

        default_exception_ret = (np.nan, np.nan, np.nan, np.nan)

        try:
            TP, FP = self.classify_answer_referred(instance)
        except Exception as e:
            print("分类回答语句|发生错误:")
            traceback.print_exc()
            print(f'当前数据:\n{json.dumps(instance, indent=2, ensure_ascii=False)}')
            return default_exception_ret
        
        try:
            FN = self.classify_ground_truth_referred(instance)
        except Exception as e:
            print("分类事实陈述|发生错误:")
            traceback.print_exc()
            print(f'当前数据:\n{json.dumps(instance, indent=2, ensure_ascii=False)}')
            return default_exception_ret
        
        TP_num = len(TP)
        FP_num = len(FP)
        FN_num = len(FN)
        print()
        try:
            precision, recall, f1_score, fnr = self.calculate_metrics(TP_num, FP_num, FN_num)
            return (precision, recall, f1_score, fnr)
        except Exception as e:
            print("计算分数|发生错误:")
            traceback.print_exc()
            print(f'当前数据:\n{json.dumps(instance, indent=2, ensure_ascii=False)}')
            return default_exception_ret


if __name__ == "__main__":
    
    instance = \
    {
        'user_input': '美国最高法院关于堕胎的裁决对全球有什么影响？',
        'reference': "美国最高法院关于堕胎的裁决具有重要的全球影响。该裁决导致在堕胎访问受到限制的州，三分之一的生育年龄女性和女孩无法获得堕胎服务。那些州的母婴健康支持也较弱，母亲死亡率较高，儿童贫困率也较高。此外，裁决的影响超出了国界，由于美国在全球的地缘政治和文化影响力，这一裁决也产生了跨国影响。全球的组织和活动家担心这一裁决可能会激励其他国家出台反堕胎的立法和政策。裁决还妨碍了某些非洲国家的进步法律改革和堕胎指南的实施。此外，该裁决在国际政策领域造成了寒蝉效应，使得反堕胎的力量能够削弱人权保护。",
        'response': "美国最高法院关于堕胎的裁决可能会产生重大影响，因为它为其他国家树立了一个先例，并影响了全球关于生育权的讨论。以下是一些潜在的影响：\n\n1. 对其他国家的影响：最高法院的裁决可以作为其他国家在处理自己的堕胎法律时的参考。它为生育权的倡导者提供了法律论据和推理，帮助他们在各自的司法管辖区内挑战限制性堕胎法律。\n\n2. 强化全球生育权运动：最高法院的有利裁决可以激励和增强全球范围内的生育权运动。它可以成为支持女性权利的活动家和组织的集结点，导致全球范围内的动员和倡导工作增加。\n\n3. 反击反堕胎运动：相反，一项限制堕胎权利的裁决可能会增强全球反堕胎运动的力量。它可能为他们的论点提供合法性，并鼓励其他国家采取类似的限制措施，可能导致现有生育权的倒退。\n\n4. 对国际援助和政策的影响：最高法院的裁决可能会影响与生育健康相关的国际援助和政策。它可能影响捐助国和组织的优先事项和资金决策，可能导致对生育权倡议的支持增加，或者相反，限制对堕胎相关服务的资助。\n\n5. 影响国际人权标准：该裁决可能有助于发展国际生育权人权标准。它可能影响现有国际人权条约和公约的解释和适用，可能加强生育权作为全球基本人权的认知。\n\n6. 全球健康影响：最高法院的裁决可能对全球健康结果产生影响，特别是在那些堕胎法律限制较严的国家。它可能影响安全合法堕胎服务的可获得性和可及性，可能导致不安全堕胎和相关健康并发症的增加。\n\n需要注意的是，具体影响将取决于最高法院裁决的性质，以及各国政府、活动家和组织在美国国内外采取的后续行动。",
        'retrieved_contexts': [
            "- 2022年，美国最高法院作出裁决，推翻了50年的判例法，取消了宪法堕胎权。\n- 这一裁决产生了巨大影响：三分之一的生育年龄女性和女孩现在生活在堕胎服务几乎完全无法获得的州。\n- 这些堕胎法律最为严格的州，母婴健康支持最为薄弱，母亲死亡率较高，儿童贫困率较高。\n- 美国最高法院的裁决还通过美国在全球的地缘政治和文化影响力，超越国界产生了影响。\n- 全球的SRR组织和活动家对这一裁决可能为其他国家的反堕胎立法和政策攻击铺路表示担忧。\n- 观察者还注意到该裁决对某些非洲国家的进步法律改革产生了影响，导致堕胎指导方针的 adoption 和执行停滞不前。\n- 该裁决在国际政策领域产生了寒蝉效应，助长了反堕胎的国家和非国家行为体破坏人权保护的势头。",
            "美国最高法院的堕胎裁决不仅在国内引发了激烈的辩论和讨论，也在全球范围内引发了广泛关注。许多国家将美国视为法律和社会问题的领导者，因此这一裁决可能会影响其他国家对堕胎的政策和态度。",
            "这一裁决还可能对国际组织和非政府组织（NGO）产生影响，尤其是那些致力于生育权和妇女健康问题的团体。根据裁决的结果，可能会出现资金、倡导工作和与美国同行的合作发生变化，进而在全球范围内引发生育正义斗争的连锁反应。"
        ]
    }

    params = {
        "model_name" : "Qwen2.5-14B-Instruct",
        "temperature": 0.2,
        "base_url": 'http://localhost:5551/v1',
        "api_key": 'EMPTY',
    }
    llm = ChatOpenAI(**params)

    embedding_params = {
        "base_url": 'http://127.0.0.1:9997/v1',
        "api_key": 'EMPTY',
    }
    embedding_model = openai.Client(**embedding_params)

    metric = Faithfulness(llm)

    score = metric.evaluate(instance)
    print(score)


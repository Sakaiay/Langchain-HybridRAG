import re
import copy
import torch
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, cast, Union

def find_minima(values, threshold):
    '''
    一组数值 values 中找到局部最小值的位置（即比邻近的值都小）。而且，这个最小值必须满足一定的阈值 threshold，
    即与相邻的值相比，最小值的差距需要超过 threshold 才被认为是有效的最小值
    '''
    minima_indices = []  
    for i in range(1, len(values) - 1):  
        # 判断当前值 values[i] 是否比前一个值 values[i - 1] 和后一个值 values[i + 1] 都小。
        # 如果是，说明 values[i] 可能是局部最小值。
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            # 差值是否大于threshold
            if (values[i - 1]-values[i]>=threshold) or (values[i + 1]-values[i]>=threshold):
                minima_indices.append(i)  
        # 处理 values[i] 小于前一个值但与后一个值相等的特殊情况
        elif values[i] < values[i - 1] and values[i] == values[i + 1]:
            if values[i - 1]-values[i]>=threshold:
                minima_indices.append(i) 
    return minima_indices


class PerplexitySplitter():
    def __init__(  
                self,   
                model_path,
                threshold,    
                device,   
                sentence_split_regex: str = r"(?<=[。！？;])\s*\n*",) -> None:  
        
        self.model_path = model_path
        self.threshold = threshold
        self.device = device
        self.sentence_split_regex = sentence_split_regex
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map=self.device)
    
    def get_ppl_batch(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        return_kv=None,
        end=None
    ):
        past_length = 0
        if end is None:
            end = input_ids.shape[1]
            
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
            
        with torch.no_grad():
            response =self.model(
                input_ids = input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = response.past_key_values  
        # (batch_size, seq_length, vocab_size)形状将是 (batch_size, seq_length-1, vocab_size)。
        # 这意味着，删除了最后一个位置的输出 logits。因为在计算交叉熵损失时，我们要预测的是每个 token 的下一个 token，
        # 因此不能包括最后一个位置的输出
        shift_logits = response.logits[..., :-1, :].contiguous()
        # 提取除了第一个input_id后面的所有id，原本的形状为(batch_size, seq_length),删除第一个位置的id，变成(batch_size, seq_length-1),因为计算交叉熵的时候，第一个单词不用预测。
        shift_labels = input_ids[..., past_length + 1 : end].contiguous()
        
        active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]
        loss = torch.nn.CrossEntropyLoss(reduction="none")(active_logits, active_labels)
        res = loss
        return (res, past_key_values) if return_kv else res
    
    def split_text(self, text: str) -> List[str]:
        
        sentences_lists = re.split(self.sentence_split_regex, text)
        
        len_sentences = []
        input_ids = torch.tensor([[]], device=self.device, dtype=torch.long)
        attention_mask = torch.tensor([[]], device=self.device, dtype=torch.long)
        for text in sentences_lists:
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            input_id = tokenized_text["input_ids"].to(self.model.device)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            len_sentences.append(input_id.shape[1])
            attention_mask_tmp = tokenized_text["attention_mask"].to(self.device)
            attention_mask = torch.cat([attention_mask, attention_mask_tmp], dim=-1)
        
        loss, past_key_values = self.get_ppl_batch( 
        input_ids,
        attention_mask,
        past_key_values=None,
        return_kv=True
        )
        first_cluster_ppl=[]
        index = 0
        for i in range(len(len_sentences)):
            if i == 0:
                first_cluster_ppl.append(loss[0:len_sentences[i]-1].mean().item())
                index += len_sentences[i] - 1
            else:
                first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
                # print(loss[index:index+len_sentences[i]])
                index += len_sentences[i]
        # 寻找最小值的索引
        minima_indices = find_minima(first_cluster_ppl, self.threshold)
        # 用来存储划分后的句子索引和句子内容
        first_chunk_indices = []
        first_chunk_sentences = []
        # 存储文本切分的边界点，包括文本的开始和结束。
        split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
        for i in range(len(split_points)-1):
            tmp_index = []
            tmp_sentence = []
            if i == 0:
                tmp_index.append(0)
                tmp_sentence.append(sentences_lists[0])
            for sp_index in range(split_points[i] + 1,split_points[i+1] + 1):
                tmp_index.append(sp_index)
                tmp_sentence.append(sentences_lists[sp_index])
            first_chunk_indices.append(tmp_index)
            first_chunk_sentences.append(tmp_sentence)
        final_chunks = []
        for sent_list in first_chunk_sentences:
            final_chunks.append(''.join(sent_list))
        # print('111',first_chunk_indices)
        return final_chunks
            
            
    def create_documents(self, texts, metadatas):
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents
    
    def split_docs(self, documents):
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

if __name__ == '__main__':
    model_path = "/data01/tqbian/modelPATH/qwen/Qwen2-7B-Instruct"
    file_path = './data/人物介绍.txt' 
    
    loader = TextLoader(file_path) 
    documents = loader.load()
    
    splitter = PerplexitySplitter(model_path=model_path, threshold=0, device='cuda:7')
    texts = splitter.split_docs(documents)
    
    print(texts)
    
    
    
"""基于相似度的切分"""
import re
import copy
import openai
import logging
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, cast, Union

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]

BreakpointThresholdType = Literal["percentile", "standard_deviation", "interquartile"]
BREAKPOINT_DEFAULTS: Dict[BreakpointThresholdType, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
}


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    if len(X) == 0 or len(Y) == 0:
        return np.array([])
    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity
    
def combine_sentences(sentences: List[dict], buffer_size: int = 1) -> List[dict]:
    """通过buffer size合并文件.
    Args:
        sentences: 需要合并的句子列表.
        buffer_size: 句子合并的数量，默认为1.
    Returns:
        合并后的句子列表
    将该句子前后的句子与它进行合并
    """
    # 循环每个句子
    for i in range(len(sentences)):
        # 创建合并的句子，用于保存将要合并的句子内容。
        combined_sentence = ""
        # 获取当前句子i之前的句子。
        for j in range(i - buffer_size, i):
            # 如果j不为负数，将当前句子（索引j的句子）添加到 combined_sentence 中。
            if j >= 0:
                combined_sentence += sentences[j]["sentence"] + " "
        combined_sentence += sentences[i]["sentence"]
        # 获取当前句子 i 之后的句子。
        for j in range(i + 1, i + 1 + buffer_size):
            # 确保不会访问超出列表长度的索引（避免越界错误）。
            if j < len(sentences):
                # 添加句子到combined_sentence
                combined_sentence += " " + sentences[j]["sentence"]

        # 将组合后的句子作为新的键 "combined_sentence" 保存到当前句子的字典中。
        sentences[i]["combined_sentence"] = combined_sentence
    return sentences

def calculate_cosine_distances(sentences: List[dict]) -> Tuple[List[float], List[dict]]:
    """计算句子之间的余弦距离，计算的是当前句子与前一个句子的相似度距离。该函数计算每个句子与其下一个句子之间的余弦距离，返回所有的距离值以及更新后的句子列表，其中每个句子的字典包含了到下一个句子的相似度距离。
    Args:
        sentences:这是一个包含句子的字典列表，每个字典包含了句子的嵌入（"combined_sentence_embedding"）和其他信息。
    Returns:
        - distances: 计算出的余弦距离列表，列表中的每个元素是一个浮动值，
                         表示当前句子与下一个句子的余弦距离（值越小表示相似度越高）。
        - sentences: 更新后的句子列表，其中每个字典都包含了当前句子和下一个句子之间的余弦距离，
                         这个信息存储在字典的键 "distance_to_next" 中。
    """
    distances = []  # 用于存储计算出的每个句子对之间的余弦距离
    # 遍历所有句子（除了最后一个句子，避免越界）
    for i in range(len(sentences) - 1):
        # 获取当前句子和下一个句子的嵌入向量
        embedding_current = sentences[i]["combined_sentence_embedding"]
        embedding_next = sentences[i + 1]["combined_sentence_embedding"]
        # 计算余弦相似度：相似度越高，返回值越接近1
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        # 将相似度转换为余弦距离：距离越小，句子越相似
        distance = 1 - similarity  # Cosine distance = 1 - cosine similarity
        # 将余弦距离添加到 distances 列表中
        distances.append(distance)
        # 将当前句子到下一个句子的余弦距离保存在字典中
        sentences[i]["distance_to_next"] = distance
    # 最后一个句子的余弦距离可以选择不计算或设为 None
    # sentences[-1]['distance_to_next'] = None  # 或者为默认值
    return distances, sentences

class SimilarityBasedSplitter():
    """通过相似度切分文档.

    来自Greg Kamradt's wonderful notebook:
    https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb

    buffer_size：决定了要与当前句子结合的前后句子的数量。例如，如果 buffer_size=1，那么当前句子将与它前面的一个句子和后面的一个句子合并。
    如果 buffer_size=2，它将与前后各两个句子合并。
    """
    def __init__(
        self,
        embedding, 
        buffer_size: int = 1, 
        breakpoint_threshold_type: BreakpointThresholdType = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        number_of_chunks: Optional[int] = None,
        sentence_split_regex: str = r"(?<=[。！？])\s*\n*",
    ):
        self.embeddings = embedding
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.number_of_chunks = number_of_chunks
        self.sentence_split_regex = sentence_split_regex
        if breakpoint_threshold_amount is None:
            self.breakpoint_threshold_amount = BREAKPOINT_DEFAULTS[
                breakpoint_threshold_type
            ]
        else:
            self.breakpoint_threshold_amount = breakpoint_threshold_amount

    def _threshold_from_clusters(self, distances: List[float]) -> float:
        """
        根据期望的块数（number_of_chunks）计算分割阈值。该方法实现了“逆百分位方法”（Inverse of percentile method）。
        参数:
            distances: 一个包含每对句子之间余弦距离的列表。
        返回:
            float: 用于分割文本的阈值，基于距离列表的百分位数。
        """
        if self.number_of_chunks is None:
            raise ValueError(
                "This should never be called if `number_of_chunks` is None."
            )
        # x1, y1 是距离列表的长度和0的百分位数
        # x2, y2 是1和100的百分位数，用于进行线性插值
        x1, y1 = len(distances), 0.0
        x2, y2 = 1.0, 100.0
        # 计算期望块数的线性插值结果，保证它在x1和x2之间
        x = max(min(self.number_of_chunks, x1), x2)
        # 通过线性插值公式计算y值，y值在0到100之间
        y = y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)
        y = min(max(y, 0), 100)   # 保证y值在[0, 100]之间
        # 使用y值作为百分位数，获取对应的阈值
        return cast(float, np.percentile(distances, y))

    def _calculate_breakpoint_threshold(self, distances: List[float]) -> float:
        """
        根据指定的阈值类型计算分割阈值（breakpoint threshold）。
        参数:
            distances: 一个浮动列表，表示句子之间的相似度距离（或差异度）。
        返回:
            float: 计算出的分割阈值。
        """
        # 如果阈值类型是百分位数，则返回指定百分位数的值
        if self.breakpoint_threshold_type == "percentile":
            return cast(
                float,
                np.percentile(distances, self.breakpoint_threshold_amount),
            )
        # 如果阈值类型是标准差，则返回基于均值加上标准差乘以某个倍数的值
        elif self.breakpoint_threshold_type == "standard_deviation":
            return cast(
                float,
                np.mean(distances)
                + self.breakpoint_threshold_amount * np.std(distances),
            )
        # 如果阈值类型是四分位距，则基于均值和四分位距来计算
        elif self.breakpoint_threshold_type == "interquartile":
            # 计算第一四分位数（25th percentile）和第三四分位数（75th percentile）
            q1, q3 = np.percentile(distances, [25, 75])
            # 计算四分位距（IQR）
            iqr = q3 - q1
            # 返回均值加上倍数的四分位距
            return np.mean(distances) + self.breakpoint_threshold_amount * iqr
        # 如果阈值类型不是以上三种，则抛出异常
        else:
            raise ValueError(
                f"Got unexpected `breakpoint_threshold_type`: "
                f"{self.breakpoint_threshold_type}"
            )

    def _calculate_sentence_distances(self, single_sentences_list: List[str]) -> Tuple[List[float], List[dict]]:
        """
        将输入的句子列表转换为句子块，并计算每对句子之间的相似度距离（余弦相似度）。
        参数:
            single_sentences_list: 输入的句子列表，每个元素是一个字符串，表示一个句子。
        返回:
            Tuple[List[float], List[dict]]:
                - List[float]: 每对句子之间的余弦相似度距离（值越大表示相似度越高，值越小表示句子差异越大）。
                - List[dict]: 每个句子的字典信息，包括句子的文本内容、索引和其他信息。
        """
        # 将每个句子包装成字典，并附上句子的索引
        _sentences = [
            {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
        ]
        # 使用 combine_sentences 函数将句子结合成句子块，考虑前后句子的合并
        sentences = combine_sentences(_sentences, self.buffer_size)
         # 对每个句子块进行embedding
        embeddings = self.embeddings.embed_documents([x["combined_sentence"] for x in sentences])
        # 将每个句子块的嵌入向量添加到句子块的字典中
        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = embeddings[i]
        # 计算每对句子块之间的余弦距离，并返回距离和句子数据
        return calculate_cosine_distances(sentences)   
     
    
     
    def split_text(self, text: str) -> List[str]:
        """
        将输入的文本根据句子间的相似度将其分割成多个块（chunks）。
        参数:
            text: 输入的长文本。
        返回:
            List[str]: 一个字符串列表，每个字符串表示文本的一个块（chunk）。
        """
        # 使用正则表达式将文本分割为单独的句子
        single_sentences_list = re.split(self.sentence_split_regex, text)

        # 如果文本只有一个句子，则直接返回原始句子列表
        if len(single_sentences_list) == 1:
            return single_sentences_list
        # 计算句子之间的相似度（距离）
        distances, sentences = self._calculate_sentence_distances(single_sentences_list)
        # 如果指定了分块数量，则根据聚类结果计算分割阈值
        if self.number_of_chunks is not None:
            breakpoint_distance_threshold = self._threshold_from_clusters(distances)
        else:
            # 否则，根据默认的分割阈值类型计算阈值
            breakpoint_distance_threshold = self._calculate_breakpoint_threshold(distances)
            
        # 获取所有距离大于阈值的句子索引，这些句子之间需要切分
        indices_above_thresh = [
            i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
        ]
        # 用于存储分割后的块（chunks）
        chunks = []
        start_index = 0
        # 根据计算出的切分点（breakpoints）将句子分割为块
        for index in indices_above_thresh:
            # 当前断点的结束索引是该句子索引
            end_index = index
            # 从当前开始索引到结束索引，取出相应的句子，合并成一个块
            group = sentences[start_index : end_index + 1]
            combined_text = " ".join([d["sentence"] for d in group])
            chunks.append(combined_text)
            start_index = index + 1
        # 最后剩余的句子形成一个新的块
        if start_index < len(sentences):
            combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
            chunks.append(combined_text)
        return chunks
    
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
        

    

    
     
            
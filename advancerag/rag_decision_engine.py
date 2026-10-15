import json
from typing import Dict, Any, Set, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.config_load import config_load
from myproject.models.models import ModelManager


class RagDecisionEngine:
    """独立RAG策略决策引擎：仅负责LLM判断预检索策略，不涉及任何检索/生成逻辑"""

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.cfg = config_load(yaml_path)
        self.decision_llm = ModelManager(self.yaml_path).init_chat_model()

        # 9种合法预检索方法白名单（包含normal表示普通检索）
        self.valid_pre_methods: Set[str] = {
            "normal",
            "multi_query",
            "decomposition",
            "parent_child",
            "self_query",
            "preview_question",
            "summary",
            "enrich",
            "hybrid_search"
        }

        self._build_decision_chain()

    def _build_decision_chain(self):
        system_prompt_text = """
你是专业RAG预检索策略决策专家，仅分析用户提问，只输出纯净JSON字符串，禁止任何多余内容：禁止注释、禁止markdown代码块、禁止解释文字、禁止多余换行。

# 全部共8种可用预检索方法，仅能从以下名称选择填入数组，无其他可选值
1. multi_query：多路查询扩展
适用场景：宽泛开放式科普、多视角行业综合问题，扩充检索词扩大召回范围
2. decomposition：问题分解检索
适用场景：复合多问题提问，一句话包含多个独立子问题，拆分后单独检索
3. parent_child：父子文档检索
适用场景：需要完整长原文上下文、长篇资料细节推导、深度原文分析
4. self_query：自查询元数据检索
适用场景：带有时间、分类、数值、标签等限定筛选条件的精准查询
5. preview_question：预览问题索引检索
适用场景：问答知识库、需要匹配同类衍生问题的场景
6. summary：摘要多向量检索
适用场景：需要概括、总结长文档内容类提问
7. enrich：文档字段增强检索
适用场景：需要补全结构化业务字段、表单信息类提问
8. hybrid_search：BM25+向量混合检索
适用场景：包含专业专有名词、关键词权重要求高的检索需求

# 输出JSON固定字段约束，仅两个字段
- use_pre_retrieval：布尔值，必须为小写true或小写false，禁止使用True/False/1/0
- pre_methods：字符串数组，数组元素只能从上述8种方法名中选择，无匹配则返回空数组[]

# 完整输出示例（必须严格参照此格式，不要添加任何额外字符）
示例1（简单事实提问）：
{{"use_pre_retrieval": false, "pre_methods": []}}

示例2（宽泛综合提问）：
{{"use_pre_retrieval": true, "pre_methods": ["multi_query"]}}

示例3（复杂复合提问，多策略组合）：
{{"use_pre_retrieval": true, "pre_methods": ["decomposition", "hybrid_search"]}}

# 决策判断参考规则
1. 简短单一纯事实提问、无多维度拓展需求：use_pre_retrieval=false，pre_methods=[]
2. 宽泛行业科普、多角度综合提问：启用multi_query
3. 一句包含多个独立小问题复合提问：启用decomposition
4. 需要完整长原文、深度细节推理：启用parent_child
5. 带有明确限定条件、筛选范围的提问：启用self_query
6. 问答知识库、匹配同类衍生问题：启用preview_question
7. 文档概括、总结类需求：启用summary
8. 需要结构化字段补全、业务表单信息：启用enrich
9. 大量专业术语、专有名词关键词检索：启用hybrid_search
10. 复杂综合性提问：可同时配置多个预检索方法
"""
        human_prompt_text = "用户原始问题：{user_query}"

        self.decision_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_text.strip()),
            ("human", human_prompt_text)
        ])
        self.decision_chain = self.decision_prompt | self.decision_llm | StrOutputParser()

    def _clean_llm_json_output(self, raw_text: str) -> str:
        if not raw_text or not raw_text.strip():
            return ""
        clean_str = raw_text.strip()
        clean_str = clean_str.replace("```json", "").replace("```", "")
        clean_str = clean_str.replace("\n", "").replace("\t", "")
        return clean_str

    def _parse_and_verify_decision(self, raw_json_text: str) -> Dict[str, Any]:
        default_decision = {
            "use_pre_retrieval": True,
            "pre_methods": ["multi_query"]
        }
        if not raw_json_text:
            return default_decision

        try:
            clean_json_str = self._clean_llm_json_output(raw_json_text)
            decision_data = json.loads(clean_json_str)

            if "use_pre_retrieval" not in decision_data:
                decision_data["use_pre_retrieval"] = default_decision["use_pre_retrieval"]
            if "pre_methods" not in decision_data or not isinstance(decision_data["pre_methods"], list):
                decision_data["pre_methods"] = []

            valid_method_list = []
            for method in decision_data["pre_methods"]:
                if isinstance(method, str) and method.strip() in self.valid_pre_methods:
                    valid_method_list.append(method.strip())
            decision_data["pre_methods"] = valid_method_list

            if not decision_data["use_pre_retrieval"]:
                decision_data["pre_methods"] = []
            if decision_data["use_pre_retrieval"] and len(decision_data["pre_methods"]) == 0:
                decision_data["pre_methods"] = ["multi_query"]

            return decision_data
        except Exception:
            return default_decision

    def decide_strategy(self, user_query: str, user_selected_methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        对外唯一入口：输入用户问题，输出校验后的决策配置
        :param user_query: 用户原始问题
        :param user_selected_methods: 用户在前端手动选择的预检索方法列表（可选）
                                      用户选择的方法会以文字方式加入到决策LLM的提示词中，让模型参考用户偏好
        """
        # 如果用户选择了normal（普通检索），直接返回不使用预检索的决策，不走LLM决策流程
        if user_selected_methods and isinstance(user_selected_methods, list) and "normal" in user_selected_methods:
            print(f"【决策引擎】用户选择普通检索(normal)，直接返回基础向量检索")
            return {"use_pre_retrieval": False, "pre_methods": []}
        
        # 构建用户偏好提示文字：把用户选择的方法以文字方式加入到提示词中
        user_hint = ""
        if user_selected_methods and isinstance(user_selected_methods, list) and len(user_selected_methods) > 0:
            # 校验方法名合法性，只保留白名单内的方法
            valid_user_methods = [m for m in user_selected_methods if m in self.valid_pre_methods]
            if valid_user_methods:
                # 把用户选择的方法以文字方式构建成提示段落，让 LLM 知道用户偏好
                user_hint = (
                    f"\n\n# 用户偏好提示\n"
                    f"用户已手动选择以下预检索方法，请在决策时优先采用这些方法（如无充分理由不要否决用户选择）：\n"
                    f"{', '.join(valid_user_methods)}\n"
                    f"请基于用户选择的方法输出决策，use_pre_retrieval 应为 true，pre_methods 应包含用户选择的方法。"
                )
                print(f"【决策引擎】用户选择预检索方法: {valid_user_methods}，已加入提示词交由LLM决策")
            else:
                print(f"【决策引擎】用户选择的方法均不合法: {user_selected_methods}，走默认LLM决策")

        try:
            # 如果有用户偏好提示，动态构建包含用户选择的 prompt
            if user_hint:
                human_prompt_with_hint = f"用户原始问题：{{user_query}}{user_hint}"
                dynamic_prompt = ChatPromptTemplate.from_messages([
                    ("system", self.decision_prompt.messages[0].prompt.template),
                    ("human", human_prompt_with_hint)
                ])
                dynamic_chain = dynamic_prompt | self.decision_llm | StrOutputParser()
                raw_output = dynamic_chain.invoke({"user_query": user_query})
            else:
                raw_output = self.decision_chain.invoke({"user_query": user_query})
            decision = self._parse_and_verify_decision(raw_output)

            # 防护：如果用户选了方法但 LLM 决策结果为空，降级直接使用用户选择
            if user_selected_methods and isinstance(user_selected_methods, list):
                valid_user_methods = [m for m in user_selected_methods if m in self.valid_pre_methods]
                if valid_user_methods and (not decision.get("pre_methods") or len(decision["pre_methods"]) == 0):
                    print(f"【决策引擎】LLM决策结果为空，降级使用用户选择: {valid_user_methods}")
                    decision["use_pre_retrieval"] = True
                    decision["pre_methods"] = valid_user_methods

            return decision
        except Exception as e:
            print(f"【决策LLM调用异常】{str(e)}，启用默认预检索策略")
            # 异常降级：如果用户选了方法，优先用用户选择；否则用默认
            if user_selected_methods and isinstance(user_selected_methods, list):
                valid_user_methods = [m for m in user_selected_methods if m in self.valid_pre_methods]
                if valid_user_methods:
                    return {
                        "use_pre_retrieval": True,
                        "pre_methods": valid_user_methods
                    }
            return {
                "use_pre_retrieval": True,
                "pre_methods": ["multi_query"]
            }

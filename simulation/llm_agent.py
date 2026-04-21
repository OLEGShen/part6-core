import json
import os
import re
from typing import Any, Dict, List

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None


class LLMDecisionError(RuntimeError):
    """Raised when an LLM decision request cannot be completed safely."""


class RiderLLMAgent:
    def __init__(self, api_key=None, base_url=None, model=None, temperature=0.3):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url or os.getenv(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model or os.getenv("DASHSCOPE_MODEL", "qwen-plus")
        self.temperature = temperature

        if OpenAI is None:
            raise LLMDecisionError("未安装 openai 包，无法启用 LLM 骑手决策。")
        if not self.api_key:
            raise LLMDecisionError("未设置 DASHSCOPE_API_KEY，无法启用 LLM 骑手决策。")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _call_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
            )
            content = (completion.choices[0].message.content or "").strip()
        except Exception as exc:
            raise LLMDecisionError(f"LLM API 调用失败: {exc}") from exc

        match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        json_text = match.group(1) if match else content

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise LLMDecisionError(f"LLM 返回内容不是有效 JSON: {content}") from exc

    def decide_work_time(self, rider_profile: Dict[str, str], decision_context: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = (
            "你在模拟一个美团骑手。你需要根据昨日表现、个人特征和当前竞争环境，"
            "决定今天的上班时间与下班时间。请务必输出 JSON。"
        )
        user_prompt = f"""
你是一个外卖骑手，人物设定如下：
{json.dumps(rider_profile, ensure_ascii=False)}

当前决策上下文如下：
{json.dumps(decision_context, ensure_ascii=False)}

请你基于以上信息做出今天的工作时间决策。

输出要求：
1. 只输出 JSON。
2. 字段必须包含：
   - thought: 简洁说明你的思考过程
   - go_to_work_time: 0 到 23 的整数
   - get_off_work_time: 0 到 23 的整数
3. 下班时间必须大于上班时间。
"""
        return self._call_json(system_prompt, user_prompt)

    def choose_orders(
        self,
        rider_profile: Dict[str, str],
        decision_context: Dict[str, Any],
        candidate_orders: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        system_prompt = (
            "你在模拟一个美团骑手。你需要根据当前位置、可接单上限、候选订单列表和个人特征，"
            "做出接单决策。请务必输出 JSON。"
        )
        user_prompt = f"""
你是一个外卖骑手，人物设定如下：
{json.dumps(rider_profile, ensure_ascii=False)}

当前决策上下文如下：
{json.dumps(decision_context, ensure_ascii=False)}

候选订单如下：
{json.dumps(candidate_orders, ensure_ascii=False)}

请从候选订单中选择你要接的订单编号。

输出要求：
1. 只输出 JSON。
2. 字段必须包含：
   - thought: 简洁说明你的接单思考过程
   - selected_order_ids: 订单 id 列表
3. 只能选择候选订单中的 id。
4. 选择数量不能超过 accept_count。
"""
        return self._call_json(system_prompt, user_prompt)

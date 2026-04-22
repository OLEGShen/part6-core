import json
import os
import re
import urllib.request
from typing import Any, Dict, List

from config import LLM_CONFIG

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None


class LLMDecisionError(RuntimeError):
    """Raised when an LLM decision request cannot be completed safely."""


class RiderLLMAgent:
    """LLM-backed rider decision maker.

    Model element mapping (Table 7-4): Agent model.
    `simulation.rider.Rider` calls this helper for work-time and order decisions.
    """

    def __init__(self, api_key=None, base_url=None, model=None, temperature=None, prompt_complexity="medium"):
        self.api_key = api_key or os.getenv(LLM_CONFIG.api_key_env)
        self.base_url = base_url or os.getenv(
            LLM_CONFIG.base_url_env, LLM_CONFIG.default_base_url
        )
        self.model = model or os.getenv(LLM_CONFIG.model_env, LLM_CONFIG.model)
        self.temperature = LLM_CONFIG.temperature if temperature is None else temperature
        self.prompt_complexity = prompt_complexity

        if not self.api_key:
            raise LLMDecisionError("未设置 DASHSCOPE_API_KEY，无法启用 LLM 骑手决策。")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if OpenAI is not None else None

    def _complexity_instruction(self) -> str:
        if self.prompt_complexity == "high":
            return "请综合考虑收益、风险、体力、订单空间分布与平台竞争状态，给出更完整的推理。"
        if self.prompt_complexity == "low":
            return "请基于最关键的 1-2 个因素快速决策。"
        return "请在收益与风险之间做平衡决策。"

    def _call_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        try:
            if self.client is not None:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                )
                content = (completion.choices[0].message.content or "").strip()
            else:
                payload = json.dumps(
                    {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": self.temperature,
                    }
                ).encode("utf-8")
                endpoint = self.base_url.rstrip("/") + "/chat/completions"
                request = urllib.request.Request(
                    endpoint,
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.loads(response.read().decode("utf-8"))
                content = (result.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
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
{self._complexity_instruction()}

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
{self._complexity_instruction()}

输出要求：
1. 只输出 JSON。
2. 字段必须包含：
   - thought: 简洁说明你的接单思考过程
   - selected_order_ids: 订单 id 列表
3. 只能选择候选订单中的 id。
4. 选择数量不能超过 accept_count。
"""
        return self._call_json(system_prompt, user_prompt)

import operator
import os
from dotenv import load_dotenv

from typing import Annotated
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import ConfigurableField

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END


load_dotenv()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")

# ロール定義
ROLES = {
    "1": {
        "name":"一般人向け健康志向トレーナー",
        "description":"健康になるための運動を目的としたトレーニング指導を行う",
        "details":"決して無理のあるトレーニングを指導せず、あくまで健康になる事を目的とした軽い運動の指導をしてください"
    },
    "2": {
        "name":"パフォーマンス向上を目指す人のためのトレーナー",
        "description":"普段の運動のパフォーマンスを向上させるためのトレーニング指導を行う",
        "details":"マラソンのタイム向上など、運動のパフォーマンスを向上させるためのトレーニングメニューを提案してくださいしてください"
    },
    "3": {
        "name":"オリンピア向けのトレーナー",
        "description":"世界一のアスリートを目指すためのトレーニング指導を行う",
        "details":"多少のケガのリスクを負ってでも、世界一を目指すためのトレーニングメニューを提案してください"
    }
}

# ステート定義
class State(BaseModel):
    query: str = Field(
        ..., description="ユーザーからの質問"
    )
    current_role: str = Field(
        default="", description="選択された回答ロール"
    )
    message: Annotated[list[str], operator.add] = Field(
        default=[], description="回答履歴"
    )
    current_judge: bool = Field(
        default=False, description="品質チェックの結果"
    )
    judgement_reason: str = Field(
        default="", description="品質チェックの判定理由"
    )

# ChatModel初期化
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    temperature=0
)
llm = llm.configurable_fields(max_tokens=ConfigurableField(id='max_tokens'))

# ノード定義
## selection_node
def selection_node(state: State) -> dict[str, Any]:
    query = state.query
    role_options = "\n".join([f"{k}. {v['name']}: {v['description']}" for k, v in ROLES.items()])
    prompt = ChatPromptTemplate.from_template(
        """質問内容に含まれるユーザーの運動履歴を判断し、最も適切な回答担当ロールを選択してください

    選択肢:
        {role_options}

    回答は選択肢の番号(1、2、または3)のみを返してください。

    質問: {query}
    """.strip()
    )

    chain = prompt | llm.with_config(configurable=dict(max_tokens=1)) | StrOutputParser()
    role_number = chain.invoke({"role_options": role_options, "query": query})

    selected_role = ROLES[role_number.strip()]["name"]
    return{"current_role": selected_role}

## answering_node
def answering_node(state: State) -> dict[str, Any]:
    query = state.query
    role = state.current_role
    role_details = "\n".join([f"- {v['name']}: {v['details']}" for v in ROLES.values()])
    prompt = ChatPromptTemplate.from_template(
        """あなたは{role}として回答してください。以下の質問に対して、あなたの役割に基づいた適切な回答し
        1週間のトレーニングメニューや食事に関するアドバイスを提供してください。

        役割の詳細:
        {role_details}

        質問: {query}
        回答:""".strip()
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"role": role, "role_details": role_details, "query": query})
    return {"message": [answer]}

## check_node

class Judgement(BaseModel):
    reason: str = Field(
        default="", description="判定理由"
    )
    judge: bool = Field(
        default=False, description="判定結果"
    )

def check_node(state: State) -> dict[str, Any]:
    query = state.query
    answer = state.message[-1]
    role = state.current_role
    prompt = ChatPromptTemplate.from_template(
        """以下の質問に対する回答を品質チェックし、問題がある場合は`False`, 問題がない場合は`True`を返してください。
        またその判断理由も説明してください。

        ユーザーからの質問: {query}
        回答: {answer}
        """.strip()
    )
    chain = prompt | llm.with_structured_output(Judgement)
    result: Judgement = chain.invoke({"role": role, "query": query, "answer": answer})
    judgement = chain.invoke({"query": query, "answer": answer})

    return {
        "current_judge": result.judge,
        "judgement_reason": result.reason
    }

# グラフの作成
workflow = StateGraph(State)

workflow.add_node("selection", selection_node)
workflow.add_node("answering", answering_node)
workflow.add_node("check", check_node)

workflow.set_entry_point("selection")

workflow.add_edge("selection", "answering")
workflow.add_edge("answering", "check")

workflow.add_conditional_edges(
    "check",
    lambda state: state.current_judge,
    {True: END, False: "answering"}
)

compiled = workflow.compile()

# question = "あまり運動習慣がなく、医者から健康のために痩せることを進められました。1年間で5kg痩せるための方法を教えてください。"
# question = "一週間で20kmほど走っています。マラソンでサブ3.5を達成したいと思っています。トレーニングメニューを教えてください。"
question = "毎日40km走っており、オリンピック代表を目指しています。世界一を目指すためのトレーニングメニューを教えてください。"

initial_state = State(query=question)
result = compiled.invoke(initial_state)

print(result)
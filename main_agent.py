import random
from dotenv import load_dotenv
from langchain_community.chat_models import ChatClovaX
from langchain.agents import Tool
from langchain_core.prompts import ChatPromptTemplate
import re
import sys
from agent_memory import AgentMemory, run_memory_tool
import final_analysis
import os
import shutil

load_dotenv(override=True)

# 1. LLM 설정
llm = ChatClovaX(model="HCX-003", max_tokens=4096)  # CLOVA X 최대 토큰 제한

# 2. 메모리 인스턴스 생성 (최대 5개, 상위 2개 유지)
agent_memory = AgentMemory(max_memory_size=5, keep_best_count=2)

# 3. Tool 함수 정의
from NaverDiscussionRAGPipeline import NaverDiscussionRAGPipeline 
from ResearchRAGPipeline import ResearchRAGPipeline
from StockPriceRAGPipeline import StockPriceRAGPipeline

# 회사명과 종목코드 매핑
COMPANY_STOCK_MAP = {
    "삼성전자": "005930",
    "sk하이닉스": "000660",
    "하이닉스": "000660",
    "sk": "000660",
    "lg": "373220",
    "네이버": "035420",
    "카카오": "035720",
    "현대차": "005380",
    "현대": "005380",
}

def extract_company_info(user_question: str):
    """사용자 질문에서 회사명과 종목코드를 추출"""
    question_lower = user_question.lower()
    
    # 회사명 찾기
    found_company = None
    for company, stock_code in COMPANY_STOCK_MAP.items():
        if company.lower() in question_lower:
            found_company = company
            break
    
    if not found_company:
        # 기본값으로 삼성전자 사용
        found_company = "삼성전자"
        stock_code = "005930"
        print(f"[경고] 질문에서 회사명을 찾을 수 없어 기본값 '{found_company}'를 사용합니다.")
    else:
        stock_code = COMPANY_STOCK_MAP[found_company]
        print(f"[회사 정보 추출] 회사: {found_company}, 종목코드: {stock_code}")
    
    return found_company, stock_code

def generate_tool_questions(company_name: str, user_question: str):
    """각 도구별로 적절한 질문 생성"""
    questions = {
        "NaverDiscussionRAGPipeline": f"{company_name}에 대한 최근 투자자 여론과 시장 관심도는 어때?",
        "ResearchRAGTool": f"최근 {company_name} 주가 분석",
        "StockPriceRAGTool": f"{company_name}의 현재 주가 상황과 최근 2달간의 가격 변화 분석"
    }
    return questions

def suggest_optimal_tools(user_question: str, agent_memory) -> str:
    """메모리에서 최적의 도구 순서 추천"""
    if len(agent_memory.memory_data["analyses"]) < 3:
        return ""
    
    # 유사한 질문들의 성공 패턴 분석
    question_keywords = set(user_question.split())
    successful_patterns = []
    
    for analysis in agent_memory.memory_data["analyses"][-20:]:  # 최근 20개 분석
        stored_keywords = set(analysis["question"].split())
        similarity = len(question_keywords & stored_keywords) / len(question_keywords | stored_keywords)
        
        if similarity > 0.2:  # 20% 이상 유사한 분석
            # 성공 여부 판단 (매수/매도/유지가 명확히 나온 경우)
            final_answer = analysis["final_answer"].lower()
            if any(keyword in final_answer for keyword in ["매수", "매도", "유지", "추천", "권장"]):
                successful_patterns.append({
                    "tools": analysis["tools_used"],
                    "similarity": similarity,
                    "answer": analysis["final_answer"]
                })
    
    if not successful_patterns:
        return ""
    
    # 가장 유사한 성공 패턴에서 도구 순서 추출
    best_pattern = max(successful_patterns, key=lambda x: x["similarity"])
    
    # 도구 순서를 문자열로 변환
    tool_order = " → ".join(best_pattern["tools"])
    
    return tool_order

def run_discussion_analysis(question: str, stock_code="005930"):
    pipeline = NaverDiscussionRAGPipeline(
        json_path="./data/discussion_comments.json",
        db_path="./chroma_langchain_db",
        collection_name="clovastudiodatas_discussion_docs"
    )
    pipeline.crawl_comments(stock_code=stock_code)
    pipeline.segment_documents()
    pipeline.embed_and_store()
    return pipeline.query_opinion(question)

def run_research_analysis(question: str):
    pipeline = ResearchRAGPipeline(
        db_path="./chroma_langchain_db",
        collection_name="clovastudiodatas_research_docs"
    )
    pipeline.extract_from_pdf_folder("./pdf_downloads")
    pipeline.segment_documents()
    pipeline.embed_and_store()
    return pipeline.query(question)

def run_stock_price_analysis(question: str, stock_code="005930"):
    pipeline = StockPriceRAGPipeline(
        db_path="./chroma_langchain_db",
        collection_name="clovastudiodatas_stock_price_docs"
    )
    pipeline.fetch_and_save(stock_code)
    pipeline.embed_and_store()
    return pipeline.query(question)





# 4. Tool 등록 (이름-함수 매핑)
tool_map = {
    "NaverDiscussionRAGPipeline": run_discussion_analysis,
    "ResearchRAGTool": run_research_analysis,
    "StockPriceRAGTool": run_stock_price_analysis,
    "MemoryTool": run_memory_tool
}

tool_desc = """
- NaverDiscussionRAGPipeline: 종토방 여론 분석 (실시간 투자자 여론)
- ResearchRAGTool: 전문가 리서치 분석
- StockPriceRAGTool: 주가 데이터 분석 (최근 2달)
- MemoryTool: 메모리 관리 (save/recall/patterns)

⚠️ Final Answer: 모든 도구 실행 완료 후에만 사용 가능한 최종 답변 도구
"""

# Observation summary 생성 함수 추가
def get_observation_summary(action_observation_log):
    summary = []
    for tool, obs in action_observation_log:
        first_line = obs.split('\n')[0]
        summary.append(f"{tool}: {first_line}")
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(summary))

# 5. 프롬프트 템플릿 (완전히 간소화된 버전)
prompt_template = ChatPromptTemplate.from_template(
"""당신은 금융 투자 분석 전문가이자 체계적인 분석 에이전트입니다.

⚠️ 반드시 아래 규칙을 지키세요:
- 한 번에 반드시 하나의 Action만 출력하세요. (절대 여러 Action을 동시에 출력하지 마세요)
- Thought, Action, Action Input 중 반드시 하나만 출력하세요.
- Observation은 직접 생성하지 마세요. (Action 실행 후, 실제 도구 실행 결과만 Observation으로 기록됩니다)
- Final Answer는 모든 도구 실행 완료 후에만 사용 가능한 최종 답변 도구입니다.
- Action, Action Input은 반드시 한 쌍으로 출력하세요.
- Action Input이 없는 Action은 무효입니다.

특히 Thought 단계에서는 아래 Observation 요약을 반드시 참고해서, 지금까지 어떤 도구를 사용했고 어떤 정보를 얻었는지 구체적으로 언급하세요.
예시: '지금까지 NaverDiscussionRAGPipeline에서 "여론 점수: 60/100, 설명: ..."을 받았고, 다음으로 전문가 의견을 분석하겠습니다.'

※ MemoryTool 사용 시 반드시 아래 지침을 따르세요:
- Action Input에는 반드시 'best', 'patterns', 'recall:질문' 등 명확한 액션을 지정하세요.
- 예시: Action Input: best / Action Input: patterns / Action Input: recall:삼성전자
- 과거 분석 결과 중 가장 효율적이었던(성과가 좋았던) 도구 조합/분석 패턴만 참고하세요. 단순히 모든 과거 분석 결과를 나열하지 마세요.

사용자 질문: {input}

사용 가능한 도구: {tool_desc}

분석 순서: 종목 토론방 → 전문가 리서치 → 주가 데이터

답변 형식:
Thought: 지금까지 사용한 도구와 얻은 정보 요약 + 다음 도구 선택 이유
Action: 도구이름
Action Input: 입력값
""")

# 6. LLM 호출 함수 (Rate Limit 방지)
def call_llm(history: str) -> str:
    import time
    import random
    
    # Rate Limit 방지를 위한 지연 (더 긴 대기 시간)
    time.sleep(5 + random.uniform(0, 3))  # 5-8초 랜덤 대기
    
    try:
        response = llm.invoke(history)
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    except Exception as e:
        if "429" in str(e):
            print("[API Rate Limit 감지] 20초 대기 후 재시도...")
            time.sleep(20)  # 20초 대기
            try:
                response = llm.invoke(history)
                if hasattr(response, 'content'):
                    return response.content
                return str(response)
            except Exception as e2:
                print(f"[재시도 실패] {e2}")
                return f"[API 오류] Rate Limit으로 인해 응답을 받을 수 없습니다. 잠시 후 다시 시도해주세요."
        else:
            raise e

# 7. ReAct 루프 구현
def react_loop(user_question: str):
    # 회사 정보 추출
    company_name, stock_code = extract_company_info(user_question)
    tool_questions = generate_tool_questions(company_name, user_question)
    
    print(f"\n[분석 대상] 회사: {company_name} (종목코드: {stock_code})")
    print(f"[도구별 질문] {tool_questions}")
    
    # 메모리 기반 최적화: 유사한 이전 분석 확인
    print("\n[메모리 기반 최적화] 유사한 이전 분석 확인 중...")
    similar_analyses = agent_memory.recall_similar_analysis(user_question, top_k=2)
    if "유사한 분석 결과가 없습니다" not in similar_analyses:
        print("[메모리 힌트] 유사한 이전 분석 발견:")
        print(similar_analyses)
        
        # 이전 분석에서 성공한 도구 순서 추천
        optimal_tools = suggest_optimal_tools(user_question, agent_memory)
        if optimal_tools:
            print(f"[최적화 제안] 추천 도구 순서: {optimal_tools}")
            # 프롬프트에 최적화 힌트 추가
            optimization_hint = f"\n메모리 기반 최적화 힌트: {optimal_tools} 순서로 분석하는 것을 권장합니다."
        else:
            optimization_hint = ""
    else:
        print("[메모리 힌트] 유사한 이전 분석이 없어 기본 전략을 사용합니다.")
        optimization_hint = ""
    
    history = prompt_template.format(input=user_question, tool_desc=tool_desc)
    if optimization_hint:
        history += optimization_hint
    print("[LLM 프롬프트]\n" + history + "\n")
    step = 1
    action_observation_log = []
    used_tools = set()  # 사용된 도구 추적
    max_steps = 15  # 토큰 제한 고려하여 스텝 수 조정
    tool_quality_check = {}  # 도구별 품질 체크
    final_answer = None
    fail_count = 0  # 연속 도구 실행 실패 카운터

    while step <= max_steps:
        print(f"\n[STEP {step}] LLM 호출 중...")
        
        # LLM 호출
        llm_output = call_llm(history)
        print(f"\n[LLM 출력]\n{llm_output}\n")

        # 첫 번째 Action~Action Input만 파싱
        action_match = re.search(r"Action\s*:\s*(\w+)", llm_output)
        action_input_match = re.search(r"Action Input\s*:?[ \t]*(.*?)(?=\n|$)", llm_output, re.DOTALL)
        if not action_match:
            print("[ERROR] LLM 출력에서 Action을 찾을 수 없습니다. LLM 출력 전체:\n", llm_output)
            break

        tool_name = action_match.group(1).strip()
        tool_input = action_input_match.group(1).strip() if action_input_match else ""

        print(f"\n[도구 실행] {tool_name} (입력: {tool_input})")
        tool_func = tool_map.get(tool_name)
        if tool_func is None:
            observation = f"[ERROR] 지원하지 않는 도구: {tool_name}"
        else:
            try:
                import time
                time.sleep(3 + random.uniform(0, 2))
                if tool_name == "MemoryTool":
                    observation = tool_func(tool_input, user_question, agent_memory)
                elif tool_name == "NaverDiscussionRAGPipeline":
                    question = tool_questions.get(tool_name, tool_input)
                    observation = tool_func(question, stock_code)
                elif tool_name == "StockPriceRAGTool":
                    question = tool_questions.get(tool_name, tool_input)
                    observation = tool_func(question, stock_code)
                elif tool_name == "ResearchRAGTool":
                    question = tool_questions.get(tool_name, tool_input)
                    observation = tool_func(question)
                else:
                    observation = tool_func(tool_input)
                if tool_name != "MemoryTool":
                    quality_score = final_analysis._evaluate_tool_quality(tool_name, observation)
                    tool_quality_check[tool_name] = quality_score
                    print(f"[품질 점수] {tool_name}: {quality_score}/10")
                    if quality_score < 5 and tool_name not in used_tools:
                        observation += f"\n[품질 경고] {tool_name}의 결과가 불충분합니다. 더 자세한 분석이 필요합니다."
            except Exception as e:
                print(f"[ERROR] 도구 실행 중 오류: {e}")
                observation = f"[분석 실패] 도구 실행에 실패했습니다. 오류: {e}"
                if tool_name != "MemoryTool":
                    tool_quality_check[tool_name] = 0

        # LLM 출력에서 Observation이 포함되어 있으면 무시 (도구 실행 결과만 Observation으로 기록)
        # 기존 LLM 출력 파싱 이후에 아래 로직을 추가
        # LLM 출력에서 Observation: ... 블록이 있으면 경고 출력 및 무시
        if re.search(r"^Observation\s*:", llm_output, re.MULTILINE):
            print("[경고] LLM 출력에 Observation이 포함되어 있어 무시합니다. 반드시 도구 실행 결과만 Observation으로 기록합니다.")
        # Observation에 도구 프롬프트 예시/지침이 섞여 들어오는지 감지 및 필터링
        def filter_prompt_leakage(obs):
            # 프롬프트/예시/지침 관련 키워드
            prompt_keywords = [
                '답변 형식', '예시', '아래 형식', '반드시', '지침', '예를 들어', '아래 예시',
                'Answer:', 'Question:', 'Context:', '아래 지침', '아래 규칙', '아래 예시를 참고',
                '아래 내용을 참고', '아래 내용을 기반', '아래 정보를 참고', '아래 정보를 기반'
            ]
            lines = obs.split('\n') if isinstance(obs, str) else [obs]
            filtered = []
            for line in lines:
                if not any(kw in line for kw in prompt_keywords):
                    filtered.append(line)
            if len(filtered) < len(lines):
                print("[경고] Observation에 프롬프트/예시/지침 관련 문구가 감지되어 자동 필터링되었습니다.")
            return '\n'.join(filtered).strip()
        # 도구 실행 결과만 Observation에 기록 (프롬프트/예시/지침 자동 필터링)
        filtered_observation = filter_prompt_leakage(observation)
        print(f"[DEBUG] observation 반환값(필터링 후):\n{filtered_observation}")
        if isinstance(filtered_observation, str) and '\n' in filtered_observation:
            history += f"\nObservation: {filtered_observation}\n"
        else:
            history += f"\nObservation: {filtered_observation}\n"
        if tool_name != "MemoryTool":
            action_observation_log.append((tool_name, observation))
            used_tools.add(tool_name)
        print(f"\n[Observation 추가됨] {observation[:100]}...")
        print(f"[사용된 도구: {used_tools}]")
        
        # 분석 완성도 체크
        completion_status = final_analysis._check_analysis_completeness(action_observation_log, tool_quality_check)
        print(f"[분석 완성도] {completion_status}")

        # 연속 2회 이상 도구 실행 실패 시 루프 종료
        if observation.startswith("[분석 실패]"):
            fail_count += 1
        else:
            fail_count = 0
        if fail_count >= 2:
            print("[ERROR] 도구 실행이 연속 2회 이상 실패하여 루프를 강제 종료합니다.")
            break

        step += 1
    
    # 최대 스텝에 도달했지만 Final Answer가 없는 경우
    if step > max_steps:
        print(f"\n[경고] 최대 스텝({max_steps})에 도달했습니다.")
        print(f"[사용된 도구: {used_tools}]")
        
        # 최종 종합 분석 실행
        if len(action_observation_log) >= 1:
            print("\n[최종 종합 분석 실행]")
            final_analysis_result = final_analysis.run_final_analysis(user_question, [obs for _, obs in action_observation_log], llm)
            final_answer = final_analysis_result.content if hasattr(final_analysis_result, 'content') else final_analysis_result
            print("\n[최종 투자 판단]")
            print(final_answer)
        
        print("\n[분석에 사용된 툴 및 Observation 요약]")
        for idx, (tool, obs) in enumerate(action_observation_log, 1):
            print(f"{idx}. {tool}: {obs[:200]}{'...' if len(obs)>200 else ''}")
    
    # 분석 결과를 메모리에 저장
    if final_answer and len(action_observation_log) >= 1:
        memory_result = agent_memory.add_analysis(
            user_question, 
            used_tools, 
            [obs for _, obs in action_observation_log], 
            final_answer
        )
        print(f"\n[메모리 저장] {memory_result}")
    
    # 토큰 사용량 최적화를 위한 히스토리 정리
    if len(history) > 10000:  # 히스토리가 너무 길어지면 정리
        print("[토큰 최적화] 히스토리 길이를 정리합니다.")
        # 최근 5개의 Action-Observation만 유지
        lines = history.split('\n')
        action_obs_sections = []
        for i, line in enumerate(lines):
            if line.startswith('Action:') or line.startswith('Observation:'):
                action_obs_sections.append(i)
        
        if len(action_obs_sections) > 10:
            keep_start = action_obs_sections[-10]
            history = '\n'.join(lines[:keep_start] + lines[keep_start:])


def clean_data_dir():
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"[초기화] {data_dir} 폴더 생성 완료")
    else:
        for f in os.listdir(data_dir):
            file_path = os.path.join(data_dir, f)
            if os.path.isfile(file_path) and f != "memory.json":
                try:
                    os.remove(file_path)
                    print(f"[초기화] {file_path} 삭제 완료")
                except Exception as e:
                    print(f"[초기화 오류] {file_path} 삭제 실패: {e}")
    chroma_db_path = "./chroma_langchain_db"
    if os.path.exists(chroma_db_path):
        try:
            shutil.rmtree(chroma_db_path)
            print(f"[초기화] {chroma_db_path} 폴더 삭제 완료")
        except Exception as e:
            print(f"[초기화 오류] {chroma_db_path} 삭제 실패: {e}")


if __name__ == "__main__":
    clean_data_dir()
    # 한 번에 한 질문만 입력받아 실행
    user_question = input("분석할 질문을 입력하세요: ")
    print(f"\n{'='*50}")
    print(f"분석 시작: {user_question}")
    print(f"{'='*50}")
    react_loop(user_question)

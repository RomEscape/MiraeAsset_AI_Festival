# LLM은 main_agent.py에서 import해서 사용

def run_final_analysis(question: str, all_observations: list, llm):
    """모든 도구의 결과를 종합하여 최종 투자 판단"""
    combined_prompt = f"""
    당신은 금융 투자 분석 전문가입니다. 다음은 삼성전자 투자 분석을 위한 모든 데이터입니다:
    
    [1. 종토방 여론 분석]
    {all_observations[0]}
    
    [2. 전문가 리서치 분석]
    {all_observations[1]}
    
    [3. 주가 데이터 분석]
    {all_observations[2]}
    
    위 데이터를 종합하여 명확한 투자 판단을 내려주세요.
    
    답변 형식:
    [투자 판단]
    매수/매도/유지 중 하나를 명확히 선택
    
    [근거 분석]
    1. 종토방 여론 근거: (구체적인 수치나 키워드 포함)
    2. 리서치 근거: (목표주가, 투자의견, 증권사명 등 포함)
    3. 주가 데이터 근거: (현재가, 추세, 변동성 등 구체적 수치 포함)
    
    [위험 요소]
    투자 시 고려해야 할 위험 요소들
    
    [기회 요소]
    투자 기회로 볼 수 있는 요소들
    
    [최종 권고]
    구체적인 투자 전략과 주의사항
    """
    return llm.invoke(combined_prompt)

def _evaluate_tool_quality(tool_name: str, observation: str) -> int:
    """도구 실행 결과의 품질을 평가 (0-10점)"""
    score = 5  # 기본 점수
    
    if tool_name == "NaverDiscussionRAGPipeline":
        if "여론" in observation and len(observation) > 100:
            score += 3
        if "투자자" in observation or "댓글" in observation:
            score += 2
    elif tool_name == "ResearchRAGTool":
        if "목표주가" in observation or "투자의견" in observation:
            score += 3
        if "증권사" in observation or "리서치" in observation:
            score += 2
    elif tool_name == "StockPriceRAGTool":
        if "현재가" in observation and "원" in observation:
            score += 3
        if "추세" in observation or "변동" in observation:
            score += 2
    
    # 에러나 빈 결과 체크
    if "ERROR" in observation or "데이터를 찾을 수 없습니다" in observation:
        score = 0
    elif len(observation) < 50:
        score = max(0, score - 3)
    
    return min(10, score)

def _check_analysis_completeness(action_observation_log: list, tool_quality_check: dict) -> str:
    """분석 완성도를 체크"""
    if len(action_observation_log) < 3:
        return f"불완전: {len(action_observation_log)}/3 도구 실행됨"
    
    total_quality = sum(tool_quality_check.values())
    avg_quality = total_quality / len(tool_quality_check) if tool_quality_check else 0
    
    if avg_quality >= 7:
        return "완료: 모든 도구가 충분한 품질로 실행됨"
    elif avg_quality >= 5:
        return f"보통: 평균 품질 {avg_quality:.1f}/10"
    else:
        return f"불충분: 평균 품질 {avg_quality:.1f}/10" 
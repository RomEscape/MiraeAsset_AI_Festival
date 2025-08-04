# LLM은 main_agent.py에서 import해서 사용

class FinalAnalysis:
    def __init__(self):
        pass
    
    def run_final_analysis(self, question: str, all_observations: list, llm, company_name: str = "삼성전자"):
        """모든 도구의 결과를 종합하여 최종 투자 판단"""
        combined_prompt = f"""
        당신은 금융 투자 분석 전문가입니다. 다음은 {company_name} 투자 분석을 위한 모든 데이터입니다:
        
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

    def evaluate_tool_quality(self, tool_name: str, observation: str) -> int:
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

    def check_analysis_completeness(self, action_observation_log: list, tool_quality_check: dict) -> str:
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
    
    def analyze_all_results(self, action_observation_log: list, tool_quality_check: dict, user_question: str, company_name: str = "삼성전자", llm=None):
        """모든 도구의 결과를 종합하여 최종 투자 판단"""
        # 관찰 결과 추출
        observations = [obs for _, obs in action_observation_log]
        
        # 품질 점수 계산
        total_quality = sum(tool_quality_check.values())
        avg_quality = total_quality / len(tool_quality_check) if tool_quality_check else 0
        
        # LLM이 없으면 기본 분석으로 대체
        if llm is None:
            return self._generate_basic_analysis(observations, avg_quality, company_name, len(action_observation_log))
        
        # LLM을 활용한 자유로운 분석
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # 분석 프롬프트 생성
        analysis_prompt = PromptTemplate(
            input_variables=["company_name", "news_analysis", "discussion_analysis", "research_analysis", "stock_analysis", "user_question"],
            template="""당신은 금융 투자 분석 전문가입니다. 다음은 {company_name}에 대한 종합 분석 데이터입니다:

[뉴스 트리거 분석]
{news_analysis}

[종토방 여론 분석]
{discussion_analysis}

[전문가 리서치 분석]
{research_analysis}

[주가 데이터 분석]
{stock_analysis}

사용자 질문: {user_question}

위 데이터를 종합하여 투자 판단을 내려주세요. 다음 형식으로 답변해주세요:

투자 판단: [매수/매도/유지 중 선택]

근거 분석:
1. 뉴스 트리거 근거: [뉴스에서 발견한 주요 신호]
2. 종토방 여론 근거: [투자자 여론 분석 결과]
3. 리서치 근거: [전문가 의견 및 목표주가]
4. 주가 데이터 근거: [기술적 분석 결과]

위험 요소:
[투자 시 고려해야 할 위험 요소들]

기회 요소:
[투자 기회로 볼 수 있는 요소들]

최종 권고:
[구체적인 투자 전략과 주의사항]

답변은 자연스럽고 전문적이어야 하며, 실제 데이터를 기반으로 한 구체적인 분석이어야 합니다."""
        )
        
        # 각 분석 결과 준비
        news_analysis = observations[0] if len(observations) > 0 else "뉴스 데이터 없음"
        discussion_analysis = observations[1] if len(observations) > 1 else "종토방 데이터 없음"
        research_analysis = observations[2] if len(observations) > 2 else "리서치 데이터 없음"
        stock_analysis = observations[3] if len(observations) > 3 else "주가 데이터 없음"
        
        # LLM 체인 실행
        try:
            analysis_chain = analysis_prompt | llm | StrOutputParser()
            result = analysis_chain.invoke({
                "company_name": company_name,
                "news_analysis": news_analysis,
                "discussion_analysis": discussion_analysis,
                "research_analysis": research_analysis,
                "stock_analysis": stock_analysis,
                "user_question": user_question
            })
            
            # 품질 점수와 도구 개수 추가
            result += f"\n\n분석 품질: {avg_quality:.1f}/10점\n실행된 도구: {len(action_observation_log)}개"
            
            return result
            
        except Exception as e:
            print(f"LLM 분석 실패: {e}")
            return self._generate_basic_analysis(observations, avg_quality, company_name, len(action_observation_log))
    
    def _generate_basic_analysis(self, observations, avg_quality, company_name, tool_count):
        """LLM 실패 시 기본 분석 생성"""
        # 데이터 내용을 기반으로 투자 성향 판단
        investment_style = "매수"
        risk_level = "보통"
        strategy = "분할 매수"
        
        # 뉴스 데이터에서 긍정/부정 신호 분석
        news_text = observations[0] if len(observations) > 0 else ""
        if any(keyword in news_text.lower() for keyword in ['급등', '상승', '호재', '성장', '돌파', '신기록']):
            investment_style = "적극 매수"
            risk_level = "높음"
            strategy = "적극적 매수"
        elif any(keyword in news_text.lower() for keyword in ['하락', '폭락', '악재', '손실', '위험', '부도']):
            investment_style = "매도"
            risk_level = "매우 높음"
            strategy = "즉시 매도"
        
        # 종토방 여론 분석
        discussion_text = observations[1] if len(observations) > 1 else ""
        if any(keyword in discussion_text.lower() for keyword in ['상승', '매수', '호재', '기대']):
            if investment_style == "매수":
                investment_style = "적극 매수"
                strategy = "적극적 매수"
        elif any(keyword in discussion_text.lower() for keyword in ['하락', '매도', '악재', '우려']):
            if investment_style == "매수":
                investment_style = "매수"
                strategy = "신중한 매수"
        
        # 리서치 분석
        research_text = observations[2] if len(observations) > 2 else ""
        if "BUY" in research_text.upper() or "매수" in research_text:
            if investment_style == "매수":
                investment_style = "적극 매수"
                strategy = "적극적 매수"
        elif "SELL" in research_text.upper() or "매도" in research_text:
            investment_style = "매도"
            strategy = "즉시 매도"
        
        # 주가 데이터 분석
        stock_text = observations[3] if len(observations) > 3 else ""
        if any(keyword in stock_text.lower() for keyword in ['상승', '돌파', '신고가', '강세']):
            if investment_style == "매수":
                investment_style = "적극 매수"
                strategy = "적극적 매수"
        elif any(keyword in stock_text.lower() for keyword in ['하락', '지지선', '약세', '저점']):
            if investment_style == "매수":
                strategy = "신중한 매수"
        
        # 위험/기회 요소 동적 생성
        risk_factors = []
        opportunity_factors = []
        
        if "급등" in news_text or "상승" in news_text:
            opportunity_factors.append("강한 상승 모멘텀")
            risk_factors.append("급등 후 조정 가능성")
        if "하락" in news_text or "악재" in news_text:
            risk_factors.append("부정적 뉴스 영향")
            opportunity_factors.append("저점 매수 기회")
        if "BUY" in research_text.upper():
            opportunity_factors.append("전문가 매수 의견")
        if "SELL" in research_text.upper():
            risk_factors.append("전문가 매도 의견")
        
        # 기본 위험/기회 요소
        if not risk_factors:
            risk_factors = ["시장 변동성", "경쟁 심화 가능성"]
        if not opportunity_factors:
            opportunity_factors = ["성장 가능성", "기술 혁신"]
        
        return f"""투자 판단: {investment_style}

근거 분석:
1. 뉴스 트리거 근거: {observations[0][:200] if len(observations) > 0 else '데이터 없음'}...
2. 종토방 여론 근거: {observations[1][:200] if len(observations) > 1 else '데이터 없음'}...
3. 리서치 근거: {observations[2][:200] if len(observations) > 2 else '데이터 없음'}...
4. 주가 데이터 근거: {observations[3][:200] if len(observations) > 3 else '데이터 없음'}...

위험 요소:
{chr(10).join([f"- {risk}" for risk in risk_factors])}

기회 요소:
{chr(10).join([f"- {opp}" for opp in opportunity_factors])}

최종 권고:
{company_name}의 종합적인 분석 결과를 바탕으로 {investment_style}를 권장합니다. 
투자 위험도: {risk_level}
투자 전략: {strategy}

분석 품질: {avg_quality:.1f}/10점
실행된 도구: {tool_count}개""" 

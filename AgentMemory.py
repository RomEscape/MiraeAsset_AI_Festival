import json
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class AgentMemory:
    def __init__(self, memory_file="./data/memory.json", max_memory_size=5, keep_best_count=2):
        self.memory_file = memory_file
        self.max_memory_size = max_memory_size
        self.keep_best_count = keep_best_count
        self.memory_data = self.load_memory()
    
    def load_memory(self):
        """메모리 파일 로드"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"analyses": []}
    
    def save_memory(self):
        """메모리 파일 저장"""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory_data, f, ensure_ascii=False, indent=2)
    
    def evaluate_analysis_quality(self, analysis):
        """분석 품질 평가 (0-10점)"""
        score = 0
        
        # 뉴스 트리거만 실행된 경우 (안정적 상태 판단)
        is_news_trigger_only = (
            len(analysis.get("tools_used", [])) == 1 and 
            "NewsRAGTool" in analysis.get("tools_used", []) and
            "안정적 상태" in analysis.get("final_answer", "")
        )
        
        if is_news_trigger_only:
            # 뉴스 트리거 시스템의 정상적인 동작
            score = 6  # 기본 점수
            final_answer = analysis.get("final_answer", "").lower()
            if "안정적" in final_answer and "추가 분석 없이" in final_answer:
                score += 2  # 명확한 판단
            return min(score, 10)
        
        # 일반적인 종합 분석의 경우
        # 도구 사용 다양성 (최대 3점)
        unique_tools = len(set(analysis.get("tools_used", [])))
        score += min(unique_tools, 3)
        
        # 관찰 결과 길이 (최대 2점)
        total_observation_length = sum(len(str(obs)) for obs in analysis.get("observations", []))
        if total_observation_length > 1000:
            score += 2
        elif total_observation_length > 500:
            score += 1
        
        # 최종 답변 품질 (최대 3점)
        final_answer = analysis.get("final_answer", "").lower()
        if any(keyword in final_answer for keyword in ["매수", "매도", "유지", "추천", "권장"]):
            score += 2
        if len(final_answer) > 200:
            score += 1
        
        # 분석 완성도 (최대 2점)
        if len(analysis.get("tools_used", [])) >= 4:
            score += 2  # 모든 도구 실행
        elif len(analysis.get("tools_used", [])) >= 2:
            score += 1  # 일부 도구 실행
        
        return min(score, 10)
    
    def update_learning_patterns(self, analysis):
        """학습 패턴 업데이트"""
        if "learning_patterns" not in self.memory_data:
            self.memory_data["learning_patterns"] = {
                "tool_performance": {},
                "company_insights": {},
                "success_patterns": [],
                "failure_patterns": []
            }
        
        # 도구별 성능 추적
        for tool in analysis.get("tools_used", []):
            if tool not in self.memory_data["learning_patterns"]["tool_performance"]:
                self.memory_data["learning_patterns"]["tool_performance"][tool] = {
                    "usage_count": 0,
                    "success_count": 0,
                    "quality_scores": []
                }
            
            tool_stats = self.memory_data["learning_patterns"]["tool_performance"][tool]
            tool_stats["usage_count"] += 1
            tool_stats["quality_scores"].append(analysis.get("quality_score", 5))
            
            # 성공 여부 판단 (품질 점수 7점 이상)
            if analysis.get("quality_score", 0) >= 7:
                tool_stats["success_count"] += 1
        
        # 회사별 인사이트 저장
        company_name = analysis.get("company_name", "")
        if company_name:
            if company_name not in self.memory_data["learning_patterns"]["company_insights"]:
                self.memory_data["learning_patterns"]["company_insights"][company_name] = {
                    "analysis_count": 0,
                    "avg_quality": 0,
                    "common_tools": [],
                    "success_patterns": []
                }
            
            company_insights = self.memory_data["learning_patterns"]["company_insights"][company_name]
            company_insights["analysis_count"] += 1
            
            # 평균 품질 점수 업데이트
            current_avg = company_insights["avg_quality"]
            new_score = analysis.get("quality_score", 5)
            company_insights["avg_quality"] = (current_avg * (company_insights["analysis_count"] - 1) + new_score) / company_insights["analysis_count"]
            
            # 성공 패턴 저장
            if analysis.get("quality_score", 0) >= 7:
                company_insights["success_patterns"].append({
                    "tools_used": analysis.get("tools_used", []),
                    "quality_score": analysis.get("quality_score", 0),
                    "timestamp": analysis.get("timestamp", "")
                })
        
        # 성공/실패 패턴 저장
        if analysis.get("quality_score", 0) >= 7:
            self.memory_data["learning_patterns"]["success_patterns"].append({
                "tools_used": analysis.get("tools_used", []),
                "company_name": analysis.get("company_name", ""),
                "quality_score": analysis.get("quality_score", 0)
            })
        else:
            self.memory_data["learning_patterns"]["failure_patterns"].append({
                "tools_used": analysis.get("tools_used", []),
                "company_name": analysis.get("company_name", ""),
                "quality_score": analysis.get("quality_score", 0)
            })
    
    def manage_memory_size(self):
        """메모리 크기 관리"""
        analyses = self.memory_data["analyses"]
        
        if len(analyses) <= self.max_memory_size:
            return
        
        # 품질 점수 계산
        for analysis in analyses:
            if "quality_score" not in analysis:
                analysis["quality_score"] = self.evaluate_analysis_quality(analysis)
        
        # 품질 점수로 정렬
        analyses.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        
        # 상위 keep_best_count개는 보존
        keep_analyses = analyses[:self.keep_best_count]
        
        # 나머지에서 최신 순으로 정렬하여 최대 크기까지 유지
        remaining_analyses = analyses[self.keep_best_count:]
        remaining_analyses.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # 최대 크기까지 유지
        max_remaining = self.max_memory_size - self.keep_best_count
        if len(remaining_analyses) > max_remaining:
            remaining_analyses = remaining_analyses[:max_remaining]
        
        # 최종 메모리 구성
        self.memory_data["analyses"] = keep_analyses + remaining_analyses
    
    def save_analysis(self, question, tools_used, final_answer, company_name="", observations=None, execution_verified=False, agent_feedback=None):
        """분석 결과 저장 (실제 도구 실행 검증 포함)"""
        if observations is None:
            observations = []
        
        # 뉴스 트리거 시스템의 정상적인 동작인지 확인
        is_news_trigger_only = (
            len(tools_used) == 1 and 
            "NewsRAGTool" in tools_used and
            "안정적 상태" in final_answer
        )
        
        # 실제 도구 실행 여부 검증 (뉴스 트리거만 실행된 경우는 허용)
        if not execution_verified and not is_news_trigger_only:
            print(f"[메모리 경고] '{company_name}' 분석이 실제 도구 실행 없이 저장되었습니다.")
            return False
        
        analysis = {
            "question": question,
            "tools_used": tools_used,
            "observations": observations,
            "final_answer": final_answer,
            "company_name": company_name,
            "timestamp": datetime.now().isoformat(),
            "quality_score": 0,  # 나중에 계산
            "agent_feedback": agent_feedback  # Agent 피드백 추가
        }
        
        # 품질 점수 계산
        analysis["quality_score"] = self.evaluate_analysis_quality(analysis)
        
        # 학습 패턴 업데이트
        self.update_learning_patterns(analysis)
        
        # 메모리에 추가
        self.memory_data["analyses"].append(analysis)
        
        # 메모리 크기 관리
        self.manage_memory_size()
        
        # 저장
        self.save_memory()
        
        return f"분석이 메모리에 저장되었습니다. (품질 점수: {analysis['quality_score']}/10)"
    
    def add_analysis(self, question, tools_used, observations, final_answer):
        """기존 호환성을 위한 메서드"""
        return self.save_analysis(question, tools_used, final_answer, observations=observations)
    
    def recall_similar_analysis(self, question, top_k=3):
        """유사한 분석 회상"""
        if len(self.memory_data["analyses"]) == 0:
            return "저장된 분석이 없습니다."
        
        # TF-IDF 벡터화
        questions = [analysis["question"] for analysis in self.memory_data["analyses"]]
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(questions)
            query_vector = vectorizer.transform([question])
            
            # 코사인 유사도 계산
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
            
            # 유사도 순으로 정렬
            similar_indices = similarities.argsort()[-top_k:][::-1]
            
            result = f"[유사 분석 회상] '{question}'와 유사한 분석 {top_k}개:\n\n"
            
            for i, idx in enumerate(similar_indices):
                analysis = self.memory_data["analyses"][idx]
                similarity = similarities[idx]
                result += f"{i+1}. 유사도: {similarity:.3f}\n"
                result += f"   질문: {analysis['question']}\n"
                result += f"   도구: {', '.join(analysis['tools_used'])}\n"
                result += f"   답변: {analysis['final_answer'][:100]}...\n"
                result += f"   품질: {analysis.get('quality_score', 0)}/10\n\n"
            
            return result
            
        except Exception as e:
            return f"유사 분석 검색 중 오류: {str(e)}"
    
    def get_recent_analyses(self, count=3):
        """최근 분석 조회"""
        if len(self.memory_data["analyses"]) == 0:
            return "저장된 분석이 없습니다."
        
        # 타임스탬프로 정렬
        sorted_analyses = sorted(
            self.memory_data["analyses"], 
            key=lambda x: x.get("timestamp", ""), 
            reverse=True
        )
        
        result = f"[최근 분석 {count}개]\n\n"
        
        for i, analysis in enumerate(sorted_analyses[:count]):
            result += f"{i+1}. 시간: {analysis.get('timestamp', 'N/A')}\n"
            result += f"   질문: {analysis['question']}\n"
            result += f"   도구: {', '.join(analysis['tools_used'])}\n"
            result += f"   품질: {analysis.get('quality_score', 0)}/10\n"
            result += f"   답변: {analysis['final_answer'][:100]}...\n\n"
        
        return result
    
    def get_best_analyses(self, count=2):
        """최고 품질 분석 조회"""
        if len(self.memory_data["analyses"]) == 0:
            return "저장된 분석이 없습니다."
        
        # 품질 점수로 정렬
        sorted_analyses = sorted(
            self.memory_data["analyses"], 
            key=lambda x: x.get("quality_score", 0), 
            reverse=True
        )
        
        result = f"[최고 품질 분석 {count}개]\n\n"
        
        for i, analysis in enumerate(sorted_analyses[:count]):
            result += f"{i+1}. 품질: {analysis.get('quality_score', 0)}/10\n"
            result += f"   질문: {analysis['question']}\n"
            result += f"   도구: {', '.join(analysis['tools_used'])}\n"
            result += f"   답변: {analysis['final_answer'][:100]}...\n\n"
        
        return result
    
    def suggest_optimal_tools(self, company_name: str = "") -> str:
        """최적 도구 순서 추천 (피드백 기반)"""
        if len(self.memory_data["analyses"]) < 2:
            return ""
        
        # 해당 회사의 최근 분석들 찾기
        company_analyses = []
        for analysis in self.memory_data["analyses"]:
            if company_name and analysis.get("company_name", "").lower() == company_name.lower():
                company_analyses.append(analysis)
            elif not company_name:  # 회사명이 없으면 모든 분석
                company_analyses.append(analysis)
        
        if not company_analyses:
            return ""
        
        # 품질 점수 기준으로 정렬
        company_analyses.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        
        # 최고 품질 분석의 도구 순서 추천
        best_analysis = company_analyses[0]
        tools_used = best_analysis.get("tools_used", [])
        
        if len(tools_used) >= 2:
            # NewsRAGTool은 항상 첫 번째로 유지
            if "NewsRAGTool" in tools_used:
                tools_used.remove("NewsRAGTool")
                recommended_order = ["NewsRAGTool"] + tools_used
            else:
                recommended_order = tools_used
            
            quality_score = best_analysis.get("quality_score", 0)
            return f"{' → '.join(recommended_order)} (품질점수: {quality_score}/10)"
        
        return ""
    
    def get_analysis_patterns(self):
        """분석 패턴 추출"""
        if len(self.memory_data["analyses"]) < 3:
            return "분석 데이터가 부족하여 패턴을 추출할 수 없습니다."
        
        # 간단한 패턴 분석
        tool_usage = {}
        common_questions = {}
        
        for analysis in self.memory_data["analyses"]:
            for tool in analysis["tools_used"]:
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
            
            question_type = "주가" if "사도 될까" in analysis["question"] else "분석"
            common_questions[question_type] = common_questions.get(question_type, 0) + 1
        
        patterns = f"[메모리 가이드] 분석 패턴 요약:\n"
        patterns += f"• 총 분석 횟수: {len(self.memory_data['analyses'])}\n"
        patterns += f"• 메모리 설정: 최대 {self.max_memory_size}개, 상위 {self.keep_best_count}개 유지\n"
        if tool_usage:
            patterns += f"• 가장 많이 사용된 도구: {max(tool_usage.items(), key=lambda x: x[1])[0]}\n"
        patterns += f"• 질문 유형 분포: {dict(common_questions)}\n"
        
        return patterns
    
    def clear_memory(self):
        """메모리 전체 삭제"""
        self.memory_data["analyses"] = []
        self.save_memory()
        return "메모리가 완전히 삭제되었습니다."
    
    def force_cleanup_memory(self):
        """현재 메모리를 설정에 맞게 강제 정리"""
        print("[강제 메모리 정리 시작]")
        self.manage_memory_size()
        self.save_memory()
        total_count = len(self.memory_data["analyses"])
        return f"메모리가 정리되었습니다. (현재 {total_count}개, 최대 {self.max_memory_size}개 유지)"
    
    def set_memory_config(self, max_size=None, keep_best=None):
        """메모리 설정 변경"""
        if max_size is not None:
            self.max_memory_size = max_size
        if keep_best is not None:
            self.keep_best_count = keep_best
        
        # 설정 변경 후 메모리 크기 재조정
        self.manage_memory_size()
        self.save_memory()
        
        return f"메모리 설정이 변경되었습니다. (최대 {self.max_memory_size}개, 상위 {self.keep_best_count}개 유지)"
    
    def run_memory_tool(self, action: str, data: str = ""):
        """메모리 저장/조회 도구 - 클래스 메서드로 통합"""
        if action == "save":
            return "메모리 저장 준비 완료. 분석 완료 후 자동 저장됩니다."
        elif action == "recall":
            return self.recall_similar_analysis(data if data else "삼성전자")
        elif action == "recent":
            try:
                count = int(data) if data else 3
                return self.get_recent_analyses(count)
            except:
                return self.get_recent_analyses(3)
        elif action == "best":
            try:
                count = int(data) if data else 2
                return self.get_best_analyses(count)
            except:
                return self.get_best_analyses(2)
        elif action == "patterns":
            return self.get_analysis_patterns()
        elif action == "clear":
            return self.clear_memory()
        elif action == "cleanup":
            return self.force_cleanup_memory()
        elif action == "config":
            try:
                # "max_size:5,keep_best:2" 형식으로 파싱
                config_parts = data.split(',')
                max_size = None
                keep_best = None
                
                for part in config_parts:
                    if 'max_size:' in part:
                        max_size = int(part.split(':')[1])
                    elif 'keep_best:' in part:
                        keep_best = int(part.split(':')[1])
                
                return self.set_memory_config(max_size, keep_best)
            except:
                return "설정 형식 오류. 예시: 'max_size:5,keep_best:2'"
        else:
            return f"지원하지 않는 메모리 액션: {action}" 

import json
import os
from datetime import datetime
import re

class AgentMemory:
    def __init__(self, memory_file="./data/memory.json", max_memory_size=5, keep_best_count=2):
        self.memory_file = memory_file
        self.max_memory_size = max_memory_size  # 최대 저장할 메모리 개수
        self.keep_best_count = keep_best_count  # 상위 성과 분석 유지 개수
        self.memory_data = self.load_memory()
    
    def load_memory(self):
        """메모리 파일 로드"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"analyses": [], "patterns": [], "last_updated": "", "memory_config": {}}
        return {"analyses": [], "patterns": [], "last_updated": "", "memory_config": {}}
    
    def save_memory(self):
        """메모리 파일 저장"""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        self.memory_data["last_updated"] = datetime.now().isoformat()
        self.memory_data["memory_config"] = {
            "max_memory_size": self.max_memory_size,
            "keep_best_count": self.keep_best_count
        }
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory_data, f, ensure_ascii=False, indent=2)
    
    def evaluate_analysis_quality(self, analysis):
        """분석 품질 평가 (점수 계산)"""
        score = 0
        
        # 1. 사용된 도구 수 (더 많은 도구 = 더 포괄적 분석)
        tool_count = len(analysis["tools_used"])
        score += tool_count * 10
        
        # 2. 최종 답변 길이 (적절한 길이 = 100-500자)
        answer_length = len(analysis["final_answer"])
        if 100 <= answer_length <= 500:
            score += 20
        elif answer_length > 500:
            score += 10
        
        # 3. 관찰 결과의 품질 (에러 메시지가 없으면 +10)
        for obs in analysis["observations"]:
            if "오류" not in obs and "error" not in obs.lower() and "Collection expecting" not in obs:
                score += 5
        
        # 4. 투자 판단의 명확성 (매수/매도/관망 등 명확한 판단)
        final_answer = analysis["final_answer"].lower()
        if any(keyword in final_answer for keyword in ["매수", "매도", "관망", "추천", "비추천"]):
            score += 15
        
        # 5. 최신성 (최근 분석일수록 높은 점수)
        try:
            timestamp = datetime.fromisoformat(analysis["timestamp"])
            days_old = (datetime.now() - timestamp).days
            if days_old <= 1:
                score += 10
            elif days_old <= 7:
                score += 5
        except:
            pass
        
        return score
    
    def manage_memory_size(self):
        """메모리 크기 관리 - 최근 N개만 유지하거나 상위 성과 분석만 유지"""
        analyses = self.memory_data["analyses"]
        
        if len(analyses) <= self.max_memory_size:
            return  # 메모리 크기가 허용 범위 내
        
        print(f"[메모리 관리] 현재 {len(analyses)}개 → 최대 {self.max_memory_size}개로 정리 중...")
        
        # 상위 성과 분석 유지 방식
        if self.keep_best_count > 0:
            # 각 분석의 품질 점수 계산
            scored_analyses = []
            for analysis in analyses:
                score = self.evaluate_analysis_quality(analysis)
                scored_analyses.append((score, analysis))
            
            # 점수순으로 정렬
            scored_analyses.sort(key=lambda x: x[0], reverse=True)
            
            # 상위 성과 분석만 유지
            best_analyses = [analysis for score, analysis in scored_analyses[:self.keep_best_count]]
            
            # 최근 분석도 일부 유지 (최근 2개)
            recent_analyses = analyses[-2:]
            
            # 중복 제거하면서 최종 메모리 구성
            final_analyses = []
            seen_timestamps = set()
            
            # 최근 분석 우선 추가
            for analysis in recent_analyses:
                if analysis["timestamp"] not in seen_timestamps:
                    final_analyses.append(analysis)
                    seen_timestamps.add(analysis["timestamp"])
            
            # 상위 성과 분석 추가
            for analysis in best_analyses:
                if analysis["timestamp"] not in seen_timestamps:
                    final_analyses.append(analysis)
                    seen_timestamps.add(analysis["timestamp"])
            
            # 최대 크기 제한 (정확히 지키기)
            if len(final_analyses) > self.max_memory_size:
                # 최근 분석 우선, 나머지는 상위 성과 순으로
                final_analyses = final_analyses[-self.max_memory_size:]
            
            print(f"[메모리 정리 완료] {len(analyses)}개 → {len(final_analyses)}개 (상위 {self.keep_best_count}개 + 최근 분석)")
            self.memory_data["analyses"] = final_analyses
            
        else:
            # 단순히 최근 N개만 유지
            self.memory_data["analyses"] = analyses[-self.max_memory_size:]
            print(f"[메모리 정리 완료] {len(analyses)}개 → {len(self.memory_data['analyses'])}개 (최근 {self.max_memory_size}개)")
    
    def add_analysis(self, question, tools_used, observations, final_answer):
        """분석 결과 저장"""
        # 중복 분석 방지: 최근 10분 내 같은 질문이 있으면 저장하지 않음
        recent_time = datetime.now().timestamp() - 600  # 10분 전
        for analysis in self.memory_data["analyses"]:
            try:
                analysis_time = datetime.fromisoformat(analysis["timestamp"]).timestamp()
                if (analysis_time > recent_time and 
                    analysis["question"] == question and 
                    len(set(analysis["tools_used"]) & set(tools_used)) >= 1):
                    print(f"[중복 분석 감지] 최근 10분 내 유사한 분석이 있어 저장을 건너뜁니다.")
                    return f"중복 분석으로 인해 저장하지 않았습니다. (최근 분석: {analysis['timestamp'][:16]})"
            except:
                continue
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "tools_used": list(tools_used),
            "observations": observations,
            "final_answer": final_answer
        }
        
        self.memory_data["analyses"].append(analysis)
        
        # 메모리 크기 관리
        self.manage_memory_size()
        
        self.save_memory()
        
        total_count = len(self.memory_data["analyses"])
        return f"분석 결과가 메모리에 저장되었습니다. (현재 {total_count}개, 최대 {self.max_memory_size}개 유지)"
    
    def recall_similar_analysis(self, question, top_k=3):
        """유사한 분석 결과 조회"""
        if not self.memory_data["analyses"]:
            return "저장된 분석 결과가 없습니다."
        
        # 간단한 키워드 매칭 (실제로는 더 정교한 유사도 계산 가능)
        question_keywords = set(question.split())
        similar_analyses = []
        
        # 전체 메모리에서 검색 (이미 크기가 제한되어 있음)
        for analysis in self.memory_data["analyses"]:
            stored_keywords = set(analysis["question"].split())
            similarity = len(question_keywords & stored_keywords) / len(question_keywords | stored_keywords)
            if similarity > 0.3:  # 30% 이상 유사
                similar_analyses.append((similarity, analysis))
        
        similar_analyses.sort(key=lambda x: x[0], reverse=True)
        
        if not similar_analyses:
            return "유사한 분석 결과가 없습니다."
        
        result = "유사한 이전 분석 결과:\n"
        for i, (similarity, analysis) in enumerate(similar_analyses[:top_k]):
            result += f"\n{i+1}. 유사도: {similarity:.1%}\n"
            result += f"   질문: {analysis['question']}\n"
            result += f"   답변: {analysis['final_answer'][:100]}...\n"
            result += f"   날짜: {analysis['timestamp'][:10]}\n"
        
        return result
    
    def get_recent_analyses(self, count=3):
        """최근 N개 분석 결과 조회"""
        if not self.memory_data["analyses"]:
            return "저장된 분석 결과가 없습니다."
        
        recent_analyses = self.memory_data["analyses"][-count:]
        
        result = f"최근 {len(recent_analyses)}개 분석 결과:\n"
        for i, analysis in enumerate(recent_analyses, 1):
            result += f"\n{i}. 질문: {analysis['question']}\n"
            result += f"   도구: {', '.join(analysis['tools_used'])}\n"
            result += f"   답변: {analysis['final_answer'][:100]}...\n"
            result += f"   날짜: {analysis['timestamp'][:10]}\n"
        
        return result
    
    def get_best_analyses(self, count=2):
        """상위 성과 분석 결과 조회"""
        if not self.memory_data["analyses"]:
            return "저장된 분석 결과가 없습니다."
        
        # 각 분석의 품질 점수 계산
        scored_analyses = []
        for analysis in self.memory_data["analyses"]:
            score = self.evaluate_analysis_quality(analysis)
            scored_analyses.append((score, analysis))
        
        # 점수순으로 정렬
        scored_analyses.sort(key=lambda x: x[0], reverse=True)
        
        result = f"상위 {min(count, len(scored_analyses))}개 성과 분석:\n"
        for i, (score, analysis) in enumerate(scored_analyses[:count], 1):
            result += f"\n{i}. 점수: {score}점\n"
            result += f"   질문: {analysis['question']}\n"
            result += f"   도구: {', '.join(analysis['tools_used'])}\n"
            result += f"   답변: {analysis['final_answer'][:100]}...\n"
            result += f"   날짜: {analysis['timestamp'][:10]}\n"
        
        return result
    
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
        
        patterns = f"분석 패턴 요약:\n"
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

def run_memory_tool(action: str, data: str = "", memory_instance=None):
    """메모리 저장/조회 도구"""
    if memory_instance is None:
        memory_instance = AgentMemory()
    
    if action == "save":
        return "메모리 저장 준비 완료. 분석 완료 후 자동 저장됩니다."
    elif action == "recall":
        return memory_instance.recall_similar_analysis(data if data else "삼성전자")
    elif action == "recent":
        try:
            count = int(data) if data else 3
            return memory_instance.get_recent_analyses(count)
        except:
            return memory_instance.get_recent_analyses(3)
    elif action == "best":
        try:
            count = int(data) if data else 2
            return memory_instance.get_best_analyses(count)
        except:
            return memory_instance.get_best_analyses(2)
    elif action == "patterns":
        return memory_instance.get_analysis_patterns()
    elif action == "clear":
        return memory_instance.clear_memory()
    elif action == "cleanup":
        return memory_instance.force_cleanup_memory()
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
            
            return memory_instance.set_memory_config(max_size, keep_best)
        except:
            return "설정 형식 오류. 예시: 'max_size:5,keep_best:2'"
    else:
        return f"지원하지 않는 메모리 액션: {action}" 

import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import chromadb
from langchain_community.embeddings import ClovaXEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import numpy as np
from dotenv import load_dotenv
import re # 텍스트 파싱을 위한 모듈 추가

class StockPriceRAGPipeline:
    def __init__(self, db_path, collection_name):
        load_dotenv(override=True)  # 환경변수 로딩 추가
        self.db_path = db_path
        self.collection_name = collection_name
        self.embeddings = ClovaXEmbeddings()
        self.client = chromadb.PersistentClient(path=db_path)
        
    def get_sise(self, code, start_time, end_time, time_from='day'):
        """주가 데이터 조회"""
        url = "https://api.finance.naver.com/siseJson.naver"
        params = {
            'symbol': code,
            'requestType': 1,
            'startTime': start_time,
            'endTime': end_time,
            'timeframe': time_from
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data_text = response.text.strip()
            # 응답 원문 출력 (디버깅)
            print("[네이버 API 응답 원문]", data_text[:300] + ("..." if len(data_text) > 300 else ""))
            # JSON 배열 형태로 반환되는 경우
            try:
                data = json.loads(data_text)
                # 네이버 API는 첫 2개(헤더/빈값) 이후가 실제 데이터인 경우가 많음
                if isinstance(data, list) and len(data) > 2 and isinstance(data[2], list):
                    keys = data[0]
                    result = []
                    for row in data[2:]:
                        if len(row) == len(keys):
                            result.append({k: v for k, v in zip(keys, row)})
                    return result
                return data
            except Exception as e:
                print(f"[경고] JSON 파싱 실패, 텍스트 파싱 시도: {e}")
                # 텍스트 파싱 (탭/쉼표/공백 등)
                lines = data_text.split('\n')
                data = []
                for line in lines:
                    line = line.strip().strip(',[]')
                    if not line or line.startswith('[') or line.startswith(']'):
                        continue
                    # 구분자: 탭, 쉼표, 공백
                    parts = [p for p in re.split(r'[\t, ]+', line) if p]
                    if len(parts) >= 6:
                        try:
                            data.append({
                                '날짜': parts[0],
                                '시가': float(parts[1].replace(',', '')),
                                '고가': float(parts[2].replace(',', '')),
                                '저가': float(parts[3].replace(',', '')),
                                '종가': float(parts[4].replace(',', '')),
                                '거래량': int(parts[5].replace(',', ''))
                            })
                        except:
                            continue
                if not data:
                    print("[경고] 텍스트 파싱 결과 데이터 없음.")
                return data
        except Exception as e:
            print(f"주가 데이터 조회 오류: {e}")
            return []

    def get_realtime_price(self, code):
        """실시간 주가 정보 조회"""
        try:
            url = f"https://polling.finance.naver.com/api/realtime/domestic/stock/{code}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if 'closePrice' in data:
                return {
                    '현재가': data['closePrice'],
                    '전일대비': data.get('changePrice', 0),
                    '등락률': data.get('changeRate', 0),
                    '거래량': data.get('accTradeVolume', 0),
                    '거래대금': data.get('accTradePrice', 0),
                    '시가총액': data.get('marketCap', 0),
                    '52주최고': data.get('high52w', 0),
                    '52주최저': data.get('low52w', 0)
                }
        except Exception as e:
            print(f"실시간 주가 조회 오류: {e}")
        return None

    def calculate_technical_indicators(self, price_data):
        """기술적 지표 계산"""
        if not price_data or len(price_data) < 20:
            return {}
        
        df = pd.DataFrame(price_data)
        df['종가'] = pd.to_numeric(df['종가'])
        df['거래량'] = pd.to_numeric(df['거래량'])
        
        # 이동평균선
        df['MA5'] = df['종가'].rolling(window=5).mean()
        df['MA20'] = df['종가'].rolling(window=20).mean()
        df['MA60'] = df['종가'].rolling(window=60).mean()
        
        # RSI 계산
        delta = df['종가'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 볼린저 밴드
        df['BB_MA20'] = df['종가'].rolling(window=20).mean()
        df['BB_STD'] = df['종가'].rolling(window=20).std()
        df['BB_UPPER'] = df['BB_MA20'] + (df['BB_STD'] * 2)
        df['BB_LOWER'] = df['BB_MA20'] - (df['BB_STD'] * 2)
        
        # MACD
        exp1 = df['종가'].ewm(span=12).mean()
        exp2 = df['종가'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_SIGNAL'] = df['MACD'].ewm(span=9).mean()
        
        # 거래량 이동평균
        df['VOLUME_MA5'] = df['거래량'].rolling(window=5).mean()
        df['VOLUME_MA20'] = df['거래량'].rolling(window=20).mean()
        
        return df.tail(1).to_dict('records')[0] if not df.empty else {}

    def fetch_and_save(self, code="005930"):
        """개선된 주가 데이터 수집 및 저장"""
        today = datetime.now()
        
        # 1. 실시간 데이터
        realtime_data = self.get_realtime_price(code)
        
        # 2. 과거 데이터 (다양한 기간)
        data_periods = [
            ("1주일", today - timedelta(days=7), today, "day"),
            ("1개월", today - timedelta(days=30), today, "day"),
            ("3개월", today - timedelta(days=90), today, "day")
        ]
        
        all_data = {}
        
        # 실시간 데이터 저장
        if realtime_data:
            all_data['실시간'] = realtime_data
        
        # 과거 데이터 수집
        for period_name, start_date, end_date, timeframe in data_periods:
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            
            period_data = self.get_sise(code, start_str, end_str, timeframe)
            if period_data:
                all_data[period_name] = period_data
                
                # 기술적 지표 계산 (1개월 데이터 기준)
                if period_name == "1개월" and len(period_data) >= 20:
                    technical_data = self.calculate_technical_indicators(period_data)
                    if technical_data:
                        all_data['기술적지표'] = technical_data
        
        # 데이터 저장
        timestamp = today.strftime("%Y%m%d_%H%M%S")
        filename = f"./data/stock_price_{code}_{timestamp}.json"
        
        os.makedirs("./data", exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2, default=str)
        
        # 텍스트 파일로도 저장 (ChromaDB용)
        text_filename = f"./data/stock_price_{code}_{timestamp}.txt"
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write(json.dumps(all_data, ensure_ascii=False, indent=2, default=str))
        
        print(f"주가 데이터 저장 완료: {filename}, {text_filename}")
        return all_data

    def segment_documents(self):
        """문서 세분화 (개선된 버전)"""
        # 이미 구조화된 데이터이므로 그대로 사용
        pass

    def embed_and_store(self):
        """임베딩 및 저장"""
        try:
            # 최신 데이터 파일 찾기
            data_files = [f for f in os.listdir("./data") if f.startswith("stock_price_") and f.endswith(".txt")]
            if not data_files:
                print("저장된 주가 데이터 파일이 없습니다.")
                return
            latest_file = max(data_files, key=lambda x: os.path.getctime(os.path.join("./data", x)))
            file_path = os.path.join("./data", latest_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 데이터를 세그먼트로 분할
            documents = []
            # 실시간 데이터
            if '실시간' in data:
                realtime = data['실시간']
                doc = Document(
                    page_content=json.dumps(realtime, ensure_ascii=False),
                    metadata={"type": "실시간", "source": latest_file}
                )
                documents.append(doc)
            # 기간별 데이터
            for period in ['1주일', '1개월', '3개월']:
                if period in data:
                    period_data = data[period]
                    if isinstance(period_data, list) and len(period_data) > 0:
                        prices = [float(item['종가']) for item in period_data if '종가' in item]
                        volumes = [int(item['거래량']) for item in period_data if '거래량' in item]
                        if prices:
                            summary = {
                                '기간': period,
                                '데이터수': len(period_data),
                                '최고가': max(prices),
                                '최저가': min(prices),
                                '평균가': sum(prices) / len(prices),
                                '변동성': (max(prices) - min(prices)) / min(prices) * 100,
                                '평균거래량': sum(volumes) / len(volumes) if volumes else 0
                            }
                            doc = Document(
                                page_content=json.dumps(summary, ensure_ascii=False),
                                metadata={"type": period, "source": latest_file}
                            )
                            documents.append(doc)
            # 기술적 지표
            if '기술적지표' in data:
                tech_data = data['기술적지표']
                doc = Document(
                    page_content=json.dumps(tech_data, ensure_ascii=False),
                    metadata={"type": "기술적지표", "source": latest_file}
                )
                documents.append(doc)
            if not documents:
                print("임베딩할 데이터가 없습니다. (주가 데이터 파싱 실패 또는 데이터 없음)")
                return
            # ChromaDB에 저장
            collection = self.client.get_or_create_collection(self.collection_name)
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            ids = [f"stock_{i}" for i in range(len(documents))]
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"ChromaDB에 {len(documents)}건 저장 완료.")
            # 저장된 결과 출력
            print("\n[저장된 임베딩 데이터 요약]")
            for doc in documents:
                print(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
        except Exception as e:
            print(f"임베딩 및 저장 오류: {e}")

    def query(self, question: str) -> str:
        """개선된 주가 데이터 쿼리"""
        try:
            collection = self.client.get_collection(self.collection_name)
            results = collection.query(
                query_texts=[question],
                n_results=10
            )
            
            if not results['documents'] or not results['documents'][0]:
                return "[주가 데이터 요약] 데이터를 찾을 수 없습니다."
            
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            
            # 데이터 타입별로 분류
            data_by_type = {}
            for i, (doc, metadata) in enumerate(zip(docs, metadatas)):
                try:
                    data = json.loads(doc)
                    data_type = metadata.get('type', 'unknown')
                    data_by_type[data_type] = data
                except:
                    continue
            
            # 종합 분석 생성
            analysis = self.generate_comprehensive_analysis(data_by_type)
            return analysis
            
        except Exception as e:
            return f"[주가 데이터 분석 오류] {e}"

    def generate_comprehensive_analysis(self, data_by_type):
        """종합적인 주가 분석 생성"""
        analysis_parts = []
        
        # 실시간 데이터 분석
        if '실시간' in data_by_type:
            realtime = data_by_type['실시간']
            analysis_parts.append(f"""
[실시간 주가 정보]
• 현재가: {realtime.get('현재가', 'N/A'):,}원
• 전일대비: {realtime.get('전일대비', 0):+,}원 ({realtime.get('등락률', 0):+.2f}%)
• 거래량: {realtime.get('거래량', 0):,}주
• 시가총액: {realtime.get('시가총액', 0):,}억원
• 52주 범위: {realtime.get('52주최저', 0):,}원 ~ {realtime.get('52주최고', 0):,}원
""")
        
        # 기간별 분석
        for period in ['1주일', '1개월', '3개월']:
            if period in data_by_type:
                period_data = data_by_type[period]
                analysis_parts.append(f"""
[{period} 분석]
• 데이터 수: {period_data.get('데이터수', 0)}일
• 가격 범위: {period_data.get('최저가', 0):,}원 ~ {period_data.get('최고가', 0):,}원
• 평균가: {period_data.get('평균가', 0):,.0f}원
• 변동성: {period_data.get('변동성', 0):.1f}%
• 평균 거래량: {period_data.get('평균거래량', 0):,}주
""")
        
        # 기술적 지표 분석
        if '기술적지표' in data_by_type:
            tech = data_by_type['기술적지표']
            analysis_parts.append(f"""
[기술적 지표]
• MA5: {tech.get('MA5', 0):,.0f}원
• MA20: {tech.get('MA20', 0):,.0f}원
• MA60: {tech.get('MA60', 0):,.0f}원
• RSI: {tech.get('RSI', 0):.1f} ({'과매수' if tech.get('RSI', 0) > 70 else '과매도' if tech.get('RSI', 0) < 30 else '중립'})
• MACD: {tech.get('MACD', 0):.2f}
• 볼린저 밴드: {tech.get('BB_LOWER', 0):,.0f}원 ~ {tech.get('BB_UPPER', 0):,.0f}원
""")
        
        # 투자 판단 근거
        current_price = data_by_type.get('실시간', {}).get('현재가', 0)
        ma20 = data_by_type.get('기술적지표', {}).get('MA20', 0)
        rsi = data_by_type.get('기술적지표', {}).get('RSI', 50)
        
        if current_price and ma20:
            price_vs_ma = ((current_price - ma20) / ma20) * 100
            trend_signal = "상승" if price_vs_ma > 0 else "하락"
            
            analysis_parts.append(f"""
[투자 판단 근거]
• 현재가 vs MA20: {price_vs_ma:+.1f}% ({trend_signal} 추세)
• RSI 신호: {'과매수' if rsi > 70 else '과매도' if rsi < 30 else '중립'}
• 기술적 신호: {'매수' if rsi < 30 and price_vs_ma < -5 else '매도' if rsi > 70 and price_vs_ma > 5 else '관망'}
""")
        
        return "\n".join(analysis_parts)

if __name__ == "__main__":
    rag = StockPriceRAGPipeline(
        db_path="./chroma_langchain_db",
        collection_name="clovastudiodatas_stock_price_docs"
    )
    rag.fetch_and_save("005930")
    rag.embed_and_store()
    print(rag.query("최근 삼성전자 주가 분석")) 

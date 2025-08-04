import os
import json
import uuid
import time
import http.client
import random
from http import HTTPStatus
from dotenv import load_dotenv
from tqdm import tqdm
from typing import List
import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import ClovaXEmbeddings
from langchain_community.chat_models import ChatClovaX
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableParallel
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException
from webdriver_manager.chrome import ChromeDriverManager
import shutil

# 정치적 키워드 상수 정의
POLITICAL_KEYWORDS = [
    # 정치 기관 및 정당
    '정치', '정부', '대통령', '국회', '여당', '야당', '민주당', '국민의힘', '자유한국당',
    
    # 정치인
    '문재인', '윤석열', '이재명', '이명박', '박근혜', '노무현', '김대중',
    
    # 정치 제도
    '정책', '법안', '입법', '행정부', '사법부', '선거', '투표', '후보', '당선', '낙선', '여론조사',
    '국정감사', '청문회', '탄핵', '국정', '국정조사', '특별법', '특별검사',
    
    # 대북 및 외교
    '북한', '김정은', '남북', '통일', '대북', '외교', '외무부',
    
    # 국방 및 안보
    '군대', '국방', '안보',
    
    # 정치 활동
    '시위', '집회', '데모', '항의', '반대',
    
    # 정치 이념
    '좌파', '우파', '진보', '보수', '이념', '사상',
    
    # 정치 관련 은어
    '트통', '도람뿌', '왕짜이밍', '빤쥬목사', '성조기', '태극기',
    '관세', '미국', '중국', '일본', '러시아', '이스라엘',
    '매국노', '기부', '협상', '만찬', '대사관', '촛불', '빤쓰교', '개딸', '고홈',
    '공황', '대폭락', '관세폭탄', '지뢰'
]

COMPANY_STOCK_MAP = {
    "삼성전자": "005930",
    "SK하이닉스": "000660",
    "카카오": "035720",
    "현대차": "005380",
}

class NaverDiscussionRAGPipeline:
    def __init__(self, json_path: str, db_path: str, collection_name: str):
        load_dotenv(override=True)
        self.json_path = json_path
        self.db_path = db_path
        self.collection_name = collection_name
        self.chunked_docs = []
        self.documents = []

        self.embedding_model = ClovaXEmbeddings(model="bge-m3")
        self.llm = ChatClovaX(model="HCX-003", max_tokens=2048)
        self.retriever = None
        self.vectorstore = None

        self._init_clova_executor()

    def _init_clova_executor(self):
        self._clova_host = "clovastudio.stream.ntruss.com"
        self._clova_api_key = os.environ["NCP_CLOVASTUDIO_API_KEY"]

    def _send_segmentation_request(self, text):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._clova_api_key}',
            'X-NCP-CLOVASTUDIO-REQUEST-ID': str(uuid.uuid4())
        }
        conn = http.client.HTTPSConnection(self._clova_host)
        request_data = {
            "postProcessMaxSize": 5000,
            "alpha": 0.0,
            "segCnt": -1,
            "postProcessMinSize": 2000,
            "text": text,
            "postProcess": True
        }
        conn.request("POST", "/testapp/v1/api-tools/segmentation", json.dumps(request_data), headers)
        response = conn.getresponse()
        status = response.status
        result = json.loads(response.read().decode("utf-8"))
        conn.close()
        if status == HTTPStatus.OK and "result" in result:
            return [' '.join(seg) for seg in result["result"]["topicSeg"]]
        else:
            raise ValueError(f"Segmentation 실패: {result}")

    def crawl_comments(self, stock_code="005930", max_scroll=20, output_path="./data/discussion_comments.json"):
        url = f"https://m.stock.naver.com/domestic/stock/{stock_code}/discussion"
        # Chrome 옵션 설정
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless')  # 헤드리스 모드
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--log-level=3')  # 에러 메시지만 표시
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])  # 로그 숨김
        chrome_options.add_argument('--silent')
        chrome_options.add_argument('--disable-logging')
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--allow-running-insecure-content')
        chrome_options.add_argument('--disable-features=VizDisplayCompositor')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        driver.get(url)
        time.sleep(1)

        comments = []
        prev_count = 0
        no_change_count = 0
        start_time = time.time()

        for _ in range(max_scroll):
            try:
                more_btn = driver.find_element(By.XPATH, '//*[@id="content"]/div[12]/div/button')
                if more_btn.is_displayed():
                    more_btn.click()
                    time.sleep(1)
            except (NoSuchElementException, ElementNotInteractableException):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)

            elements = driver.find_elements(By.XPATH, '//li[contains(@class, "DiscussionPostWrapper_article")]')
            if len(elements) == prev_count:
                no_change_count += 1
                if no_change_count >= 3:
                    break
            else:
                no_change_count = 0
            prev_count = len(elements)

            if time.time() - start_time > 60:  # 60초로 증가
                break

        for elem in elements:
            try:
                title = elem.find_element(By.XPATH, './div[2]/strong').text.strip()
            except:
                title = ""
            try:
                body = elem.find_element(By.XPATH, './div[2]/p').text.strip()
            except:
                body = ""
            if title or body:
                comments.append({"content": f"{title}\n{body}".strip()})

        driver.quit()

        # 정치적 내용 필터링 및 종목 관련성 검증
        filtered_comments = self._filter_relevant_comments(comments, stock_code)
        
        # 240개를 목표로 하되, 부족하면 원본에서 추가
        target_count = 240
        if len(filtered_comments) < target_count:
            print(f"[댓글 부족] 필터링된 댓글 {len(filtered_comments)}개, 목표 {target_count}개")
            # 원본에서 정치적 키워드만 제외하고 추가
            backup_comments = []
            for comment in comments:
                content = comment.get("content", "").lower()
                has_political = any(keyword in content for keyword in POLITICAL_KEYWORDS)
                if not has_political and len(content.strip()) >= 5:
                    # 추가 검증: 정치적 내용이 전혀 없는 경우만 포함
                    political_count = sum(1 for keyword in POLITICAL_KEYWORDS if keyword in content)
                    if political_count == 0:
                        backup_comments.append(comment)
            
            # 중복 제거하면서 추가
            existing_contents = {comment.get("content", "") for comment in filtered_comments}
            for comment in backup_comments:
                if comment.get("content", "") not in existing_contents:
                    filtered_comments.append(comment)
                    if len(filtered_comments) >= target_count:
                        break
            
            print(f"[댓글 보충] 최종 {len(filtered_comments)}개 확보")
        
        # 최대 240개로 제한
        if len(filtered_comments) > target_count:
            filtered_comments = filtered_comments[:target_count]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered_comments, f, ensure_ascii=False, indent=2)
        print(f"원본 댓글 {len(comments)}개 중 종목 관련 댓글 {len(filtered_comments)}개 저장 완료: {output_path}")
        
        return filtered_comments

    def _filter_relevant_comments(self, comments, stock_code="005930"):
        """정치적 내용을 필터링하고 종목에 직접적인 영향을 주는 의견만 선별"""
        
        filtered_comments = []
        
        for comment in comments:
            content = comment.get("content", "").lower()
            
            # 정치적 키워드가 포함된 경우 제외
            has_political = any(keyword in content for keyword in POLITICAL_KEYWORDS)
            if has_political:
                continue
            
            # 정치적 키워드가 없으면 포함 (은어나 새로운 표현도 포착 가능)
            if len(content.strip()) >= 10:  # 최소 10자 이상
                # 추가 검증: 정치적 내용이 전혀 없는 경우만 포함
                political_count = sum(1 for keyword in POLITICAL_KEYWORDS if keyword in content)
                if political_count == 0:
                    filtered_comments.append(comment)
        
        print(f"[필터링 결과] 원본 {len(comments)}개 → 정치적 내용 제외 후 {len(filtered_comments)}개")
        
        return filtered_comments

    def _load_documents(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            comments = json.load(f)
        docs = []
        for i, item in enumerate(comments):
            docs.append(Document(
                page_content=item.get("content", ""),
                metadata={
                    "source": os.path.basename(self.json_path),
                    "id": str(i)
                }
            ))
        return docs

    def segment_documents(self):
        docs = self._load_documents()
        self.chunked_docs = []
        group_size = 10
        for i in tqdm(range(0, len(docs), group_size), desc="Segmentation 요청 처리"):
            group = docs[i:i+group_size]
            merged_discussion_text = "\n\n".join([d.page_content for d in group])
            merged_ids = [d.metadata.get("id") for d in group]
            try:
                result_data = self._send_segmentation_request(merged_discussion_text)
                for paragraph in result_data:
                    self.chunked_docs.append({
                        "page_content": paragraph,
                        "metadata": {"source_ids": merged_ids}
                    })
                time.sleep(1)
            except Exception as e:
                print(f"Segmentation 실패: {e}")
                time.sleep(3)
        
        # 파일 저장 제거 - discussion_comments.json만 유지
        print("[세그멘테이션] 완료 - discussion_comments.json만 생성됨")

    def embed_and_store(self):
        print("[임베딩] 시작")
        if not self.chunked_docs:
            raise ValueError("세그멘테이션이 먼저 실행되어야 합니다.")

        print(f"[임베딩] chunked_docs 개수: {len(self.chunked_docs)}")
        self.documents = []
        for item in self.chunked_docs:
            self.documents.append(Document(
                page_content=item["page_content"],
                metadata={
                    "source": item["metadata"],
                    "id": str(uuid.uuid4())
                }
            ))
        print(f"[임베딩] documents 생성 완료: {len(self.documents)}개")

        print("[임베딩] ChromaDB 클라이언트 초기화 시작")
        client = chromadb.PersistentClient(path=self.db_path)
        print("[임베딩] ChromaDB 클라이언트 초기화 완료")
        
        # 기존 컬렉션 삭제 후 새로 생성
        print(f"[임베딩] 컬렉션 '{self.collection_name}' 처리 시작")
        try:
            client.delete_collection(name=self.collection_name)
            print("[임베딩] 기존 컬렉션 삭제 완료")
        except Exception:
            print("[임베딩] 기존 컬렉션 없음")
            pass  # 컬렉션이 없으면 무시
        client.create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
        print("[임베딩] 새 컬렉션 생성 완료")
        print("[임베딩] Chroma vectorstore 초기화 시작")
        self.vectorstore = Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model
        )
        print("[임베딩] Chroma vectorstore 초기화 완료")

        print("[임베딩] 텍스트 처리 시작")
        texts, metadatas = [], []
        for i, doc in enumerate(self.documents):
            original = doc.metadata
            source_ids = original.get("source", {}).get("source_ids", [])

            # 핵심 수정: 문자열 보장 + 길이 제한 + 널문자 제거
            text = str(doc.page_content).replace("\x00", "").strip()[:8000]
            
            if not text:
                continue  # 빈 텍스트는 건너뜀

            flattened_metadata = {
                "source_ids": ",".join(source_ids) if isinstance(source_ids, list) else str(source_ids),
                "id": original.get("id", "")
            }

            texts.append(text)
            metadatas.append(flattened_metadata)
            
            if i % 5 == 0:  # 5개마다 진행상황 출력
                print(f"[임베딩] 텍스트 처리 진행: {i+1}/{len(self.documents)}")

        print(f"[임베딩] 텍스트 처리 완료: {len(texts)}개")
        print("[임베딩] ChromaDB에 텍스트 추가 시작")
        
        # 배치 단위로 나누어 처리 (임베딩 타임아웃 방지)
        batch_size = 5
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            print(f"[임베딩] 배치 {i//batch_size + 1} 처리 중: {len(batch_texts)}개")
            self.vectorstore.add_texts(texts=batch_texts, metadatas=batch_metadatas)
            print(f"[임베딩] 배치 {i//batch_size + 1} 완료")
        
        print("[임베딩] ChromaDB에 텍스트 추가 완료")
        print(f"{len(texts)}개 문서가 ChromaDB에 저장되었습니다.")

    def query_opinion(self, question: str) -> str:
        if self.vectorstore is None:
            raise ValueError("임베딩이 먼저 수행되어야 합니다.")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        prompt = PromptTemplate.from_template(
            '''당신은 금융 전문가이자 투자 심리 분석가입니다.
아래에 제공된 문맥(context)은 특정 종목에 대한 최근 투자자들의 댓글과 여론입니다.
당신의 임무는 이 문맥에 포함된 여론을 객관적으로 요약하고, 투자 심리와 감정적 분위기를 평가하는 것입니다.

답변 형식(반드시 아래 형식을 지키세요):
여론 점수: (0~100, 부정적일수록 0, 긍정적일수록 100)
설명: 왜 이런 점수가 나왔는지, 근거가 되는 여론/심리/표현을 요약

⚠️ 반드시 실제 도구 실행 결과만 사용하세요. 예시를 복사하지 마세요.

# Question:
{question}

# Context:
{context}

# Answer:'''
        )

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | self.llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        result = rag_chain_with_source.invoke(question)
        
        # 원본 댓글 개수 정보 추가
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                original_comments = json.load(f)
            original_count = len(original_comments)
        except:
            original_count = "알 수 없음"
        
        # 결과에 원본 댓글 개수 정보 포함
        final_result = f"종목 토론방 댓글 {original_count}개를 수집하여 RAG 점수를 계산하였습니다.\n\nResult:\n{result['answer']}"
        return final_result

def main():
    company_name = input("분석할 회사명을 입력하세요: ").strip()
    stock_code = COMPANY_STOCK_MAP.get(company_name)
    if not stock_code:
        print("해당 회사의 종목코드가 등록되어 있지 않습니다.")
        return

    json_path = f"./data/{company_name}_discussion_comments.json"
    collection_name = f"{company_name}_discussion_docs"
    pipeline = NaverDiscussionRAGPipeline(
        json_path=json_path,
        db_path="./chroma_langchain_db",
        collection_name=collection_name
    )
    pipeline.crawl_comments(stock_code=stock_code, output_path=json_path)
    pipeline.segment_documents()
    pipeline.embed_and_store()
    query = f"{company_name}에 대한 최근 여론이 어때?"
    answer = pipeline.query_opinion(query)
    print("\n질문:", query)
    print("분석 결과:\n", answer)

if __name__ == "__main__":
    main()
